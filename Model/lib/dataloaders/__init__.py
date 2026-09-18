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

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .ISIC2018Dataset import ISIC2018Dataset


def get_dataloader(opt):
    """
    get dataloader
    Args:
        opt: params dict
    Returns:
    """

    if opt["dataset_name"] == "ISIC-2018":
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


def get_test_dataloader(opt):
    """
    get test dataloader
    :param opt: params dict
    :return:
    """
    if opt["dataset_name"] == "ISIC-2018":
        valid_set = ISIC2018Dataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    return valid_loader
