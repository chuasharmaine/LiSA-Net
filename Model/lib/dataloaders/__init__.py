# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 16:56
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .ISIC2018Dataset import ISIC2018Dataset

# helper function to extract class labels for a list of image names from the dataset
def _labels_for(dataset, names):
    return np.asarray([int(np.argmax(dataset.cls_labels_dict[name])) for name in names])

# for k-fold cross-validation, stratified sampling to ensure class distribution is preserved in each fold
def _stratified_fold_names(dataset, folds, fold, seed):
    names = np.asarray(dataset.image_names)
    labels = _labels_for(dataset, names)
    rng = np.random.RandomState(seed)
    valid_indices = []
    for class_index in range(dataset.opt["cls_classes"]):
        class_indices = np.flatnonzero(labels == class_index)
        if len(class_indices) < folds:
            raise ValueError(f"Class {class_index} has only {len(class_indices)} samples for {folds}-fold CV")
        rng.shuffle(class_indices)
        valid_indices.extend(np.array_split(class_indices, folds)[fold])
    valid_indices = np.asarray(sorted(valid_indices), dtype=int)
    is_valid = np.zeros(len(names), dtype=bool)
    is_valid[valid_indices] = True
    return names[~is_valid].tolist(), names[is_valid].tolist()

# set class weights for the current fold based on the training set, using tempered inverse square root of class counts
def _set_fold_class_weights(opt, train_set):
    labels = _labels_for(train_set, train_set.image_names)
    counts = np.bincount(labels, minlength=opt["cls_classes"])
    if np.any(counts == 0):
        raise ValueError(f"Training fold has empty classes: {np.flatnonzero(counts == 0).tolist()}")
    weights = 1.0 / np.sqrt(counts.astype(np.float64))
    weights /= weights.mean()
    opt["class_weight"] = weights.tolist()
    print("Fold training class counts:", counts.tolist())
    print("Tempered class weights:", [round(value, 6) for value in weights])

def get_dataloader(opt):
    """
    get dataloader
    Args:
        opt: params dict
    Returns:
    """
    # edited to support cross-validation and weighted sampling for class imbalance
    if opt["dataset_name"] == "ISIC-2018":
        cv_folds = opt.get("cv_folds", 0)
        cv_fold = opt.get("cv_fold")
        if cv_folds:
            if not (opt.get("segmentation") and opt.get("classification")):
                raise ValueError("Cross-validation currently supports multitask training only")
            if cv_folds < 2 or cv_fold is None or not 0 <= cv_fold < cv_folds:
                raise ValueError("A valid zero-based cv_fold is required when cv_folds is enabled")
            source_modes = ("train", "valid")
            pool_set = ISIC2018Dataset(opt, mode="valid", source_modes=source_modes)
            train_names, valid_names = _stratified_fold_names(
                pool_set, cv_folds, cv_fold, opt["seed"]
            )
            train_set = ISIC2018Dataset(
                opt, mode="train", source_modes=source_modes, image_names=train_names
            )
            valid_set = ISIC2018Dataset(
                opt, mode="valid", source_modes=source_modes, image_names=valid_names
            )
            test_pattern = os.path.join(opt["dataset_path"], "multitask", "test", "images", "*.jpg")
            test_names = {os.path.splitext(os.path.basename(path))[0] for path in glob.glob(test_pattern)}
            development_names = set(pool_set.image_names)
            overlap = development_names & test_names
            if overlap:
                raise ValueError(f"Development/test leakage: {len(overlap)} shared image IDs")
            print(
                f"Stratified CV fold {cv_fold + 1}/{cv_folds}: "
                f"train={len(train_set)}, valid={len(valid_set)}, held-out test={len(test_names)}; no test overlap"
            )
            if opt.get("class_weighting") == "sqrt_inverse":
                _set_fold_class_weights(opt, train_set)
            else:
                opt["class_weight"] = None
        else:
            train_set = ISIC2018Dataset(opt, mode="train")
            valid_set = ISIC2018Dataset(opt, mode="valid")

        if (opt.get("classification_data_source") == "multitask"
                and opt.get("classification") and not opt.get("segmentation")):
            test_pattern = os.path.join(opt["dataset_path"], "multitask", "test", "images", "*.jpg")
            test_names = {os.path.splitext(os.path.basename(path))[0] for path in glob.glob(test_pattern)}
            split_names = {
                "train": set(train_set.image_names),
                "valid": set(valid_set.image_names),
                "test": test_names,
            }
            if any(not names for names in split_names.values()):
                raise ValueError("Multitask train, valid, and test image splits must all be nonempty")
            for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
                overlap = split_names[left] & split_names[right]
                if overlap:
                    raise ValueError(f"Multitask {left}/{right} image leakage: {len(overlap)} shared IDs")
            print("Classification data: multitask split (train={}, valid={}, held-out test={}; no overlapping image IDs)".format(
                *(len(split_names[name]) for name in ("train", "valid", "test"))))

        # weighted sampling
        #  - addressing class imbalance by increasing the probability of selecting samples from underrepresented classes
        #  - images from smaller classes are more likely to be picked in each batch
        if opt.get("classification", False) and opt.get("oversample", False):
            # calculate weights from the actual filtered training list
            train_labels = [
                int(torch.tensor(train_set.cls_labels_dict[name]).argmax().item())
                for name in train_set.image_names
            ]
            train_counts = torch.bincount(
                torch.tensor(train_labels), minlength=opt["cls_classes"]
            )
            if torch.any(train_counts == 0):
                missing_classes = torch.where(train_counts == 0)[0].tolist()
                raise ValueError(f"Training split has no samples for classification classes: {missing_classes}")
            class_weights = 1.0 / train_counts.float()
            sampler_generator = torch.Generator()
            sampler_generator.manual_seed(opt["seed"])

            # assign a weight to every sample based on its class label
            sample_weights = [class_weights[label] for label in train_labels]

            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.float),
                num_samples=len(sample_weights),
                replacement=True,
                generator=sampler_generator
            )

            train_loader = DataLoader(train_set, batch_size=opt["batch_size"], sampler=sampler, num_workers=opt["num_workers"], pin_memory=True)

        else:
            train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    opt["steps_per_epoch"] = len(train_loader)

    return train_loader, valid_loader

# added function to get a test dataloader for evaluation, supporting cross-validation and multitask classification
def get_test_dataloader(opt):
    """
    get test dataloader
    :param opt: params dict
    :return:
    """
    if opt["dataset_name"] == "ISIC-2018":
        if opt.get("cv_folds"):
            source_modes = ("train", "valid")
            pool_set = ISIC2018Dataset(opt, mode="valid", source_modes=source_modes)
            _, valid_names = _stratified_fold_names(
                pool_set, opt["cv_folds"], opt["cv_fold"], opt["seed"]
            )
            valid_set = ISIC2018Dataset(
                opt, mode="valid", source_modes=source_modes, image_names=valid_names
            )
            print(
                f"Evaluating stratified CV fold {opt['cv_fold'] + 1}/{opt['cv_folds']}: "
                f"valid={len(valid_set)}"
            )
        else:
            valid_set = ISIC2018Dataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    return valid_loader