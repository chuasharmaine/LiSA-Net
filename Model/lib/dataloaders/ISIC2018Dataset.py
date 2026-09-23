# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/29 01:02
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import glob
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import lib.utils as utils
import lib.transforms.two as my_transforms

from collections import defaultdict
import random


class ISIC2018Dataset(Dataset):
    """
    load ISIC 2018 dataset
    """

    def __init__(self, opt, mode, source_modes=None, image_names=None):
        """
        initialize ISIC 2018 dataset
        :param opt: params dict
        :param mode: train/valid
        """
        super(ISIC2018Dataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.segmentation = opt["segmentation"]
        self.classification = opt["classification"]

        if self.segmentation and self.classification:
            data_source = "multitask"
        elif self.segmentation:
            data_source = "segmentation"
        else:
            data_source = opt.get("classification_data_source", "classification")
            if data_source not in ("classification", "multitask"):
                raise ValueError(f"Unknown classification data source: {data_source}")

        source_modes = source_modes or (mode,)
        self.root = os.path.join(opt["dataset_path"], data_source, source_modes[0])
        self.image_root_dict = {}
        discovered_names = []
        for source_mode in source_modes:
            source_root = os.path.join(opt["dataset_path"], data_source, source_mode)
            for path in sorted(glob.glob(os.path.join(source_root, "images", "*.jpg"))):
                name = os.path.splitext(os.path.basename(path))[0]
                if name in self.image_root_dict:
                    raise ValueError(f"Duplicate image ID across source splits: {name}")
                self.image_root_dict[name] = source_root
                discovered_names.append(name)

        if image_names is None:
            self.image_names = discovered_names
        else:
            missing = sorted(set(image_names) - set(self.image_root_dict))
            if missing:
                raise ValueError(f"Selected images are missing from the source splits: {missing[:5]}")
            self.image_names = list(image_names)

        self.cls_labels_dict = {}
        # Classification
        if self.classification:
            for source_mode in source_modes:
                label_csv = os.path.join(opt["dataset_path"], data_source, source_mode, "labels.csv")
                with open(label_csv, "r") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        image_id = row[0]
                        labels = list(map(float, row[1:]))
                        previous = self.cls_labels_dict.get(image_id)
                        if previous is not None and previous != labels:
                            raise ValueError(f"Conflicting labels for image ID: {image_id}")
                        self.cls_labels_dict[image_id] = labels

            self.image_names = [n for n in self.image_names if n in self.cls_labels_dict]

        # duplicating minority-class samples
        #  - to increase its representation in the training set
        #  - a limit of not exceeding 2x its original count is applied to avoid excessive duplication
        #  *note: this only repeats the image names in the dataset list
        if self.opt.get("oversample", False) and self.mode == "train" and self.classification:
            class_images = defaultdict(list)

            for name in self.image_names:
                label = np.argmax(self.cls_labels_dict[name])
                class_images[label].append(name)
            target = max(len(v) for v in class_images.values())

            max_multiplier = 2
            balanced_names = []
            for label, images in class_images.items():
                balanced_names.extend(images)
                desired = min(target, len(images) * max_multiplier)
                if len(images) < desired:
                    extra = random.choices(images, k=desired - len(images))
                    balanced_names.extend(extra)
                
            self.image_names = balanced_names

        # transforms
        self.transforms_dict = {
            "train": my_transforms.Compose([
                my_transforms.RandomResizedCrop(self.opt["resize_shape"], scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'),
                my_transforms.ColorJitter(brightness=self.opt["color_jitter"], contrast=self.opt["color_jitter"], saturation=self.opt["color_jitter"], hue=0),
                my_transforms.RandomGaussianNoise(p=self.opt["augmentation_p"]),
                my_transforms.RandomHorizontalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomVerticalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomRotation(self.opt["random_rotation_angle"]),
                my_transforms.Cutout(p=self.opt["augmentation_p"], value=(0, 0)),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ]),
            "valid": my_transforms.Compose([
                my_transforms.Resize(self.opt["resize_shape"]),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ])
        }

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        
        image_root = self.image_root_dict[image_name]
        img_path = os.path.join(image_root, "images", image_name + ".jpg")
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"cannot find: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = None
        if self.segmentation:
            mask_path = os.path.join(image_root, "masks", image_name + "_segmentation.png")
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                raise RuntimeError(f"Missing mask: {mask_path}")
            mask[mask == 255] = 1 
        
        # apply transforms
        if self.segmentation and self.classification:
            image, mask = self.transforms_dict[self.mode](image, mask)
        elif self.segmentation:
            image, mask = self.transforms_dict[self.mode](image, mask)
        elif self.classification:
            image, _ = self.transforms_dict[self.mode](image, np.zeros_like(image[:,:,0]))

        if self.segmentation:
            mask = (mask > 0).float()

        label = None
        if self.classification:
            label_list = self.cls_labels_dict[image_name]
            label = torch.tensor(label_list, dtype=torch.float32)
            label = torch.argmax(label).long()

        if self.segmentation and self.classification:
            return image, mask, label
        elif self.segmentation:
            return image, mask
        elif self.classification:
            return image, label