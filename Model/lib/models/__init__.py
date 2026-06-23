# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 16:57
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch
import torch.optim as optim

import lib.utils as utils

from .UNet import UNet
from .EGEUNet import EGEUNet
from .PMFSNet import PMFSNet
from .LiSANet import LiSANet

from .ResNet50 import ResNet50
from .DenseNet121 import DenseNet121
from .EfficientNetV2 import EfficientNetV2
from .MobileNetV3 import MobileNetV3

from .BreastCancerMT import BreastCancerMT
from .MBDCNN import MBDCNN

from .LiSANetMT import LiSANetMT




def get_model_optimizer_lr_scheduler(opt):
    # initialize model
    if opt["dataset_name"] == "ISIC-2018":

        # segmentation only models
        if opt["model_name"] == "UNet":
            model = UNet(n_channels=opt["in_channels"], n_classes=opt["seg_classes"])

        elif opt["model_name"] == "EGEUNet":
            model = EGEUNet(input_channels=opt["in_channels"], num_classes=opt["seg_classes"])

        elif opt["model_name"] == "PMFSNet":
            model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["seg_classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"])

        elif opt["model_name"] == "LiSANet":
            model = LiSANet(in_channels=opt["in_channels"], out_channels=opt["seg_classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"])


        # classification only models
        elif opt["model_name"] == "ResNet50":
            model = ResNet50(num_classes=opt["cls_classes"], pretrained=True)

        elif opt["model_name"] == "DenseNet121":
            model = DenseNet121(num_classes=opt["cls_classes"], pretrained=True)

        elif opt["model_name"] == "EfficientNetV2":
            model = EfficientNetV2(num_classes=opt["cls_classes"], pretrained=True)

        elif opt["model_name"] in ["MobileNetV3", "mobilenetv3_large_100"]:
            model = MobileNetV3(num_classes=opt["cls_classes"], pretrained=True)


        # multitask models
        elif opt["model_name"] == "BreastCancerMT":
            model = BreastCancerMT(
                in_channels=opt["in_channels"],
                seg_out_channels=opt["seg_classes"],
                cls_out_channels=opt["cls_classes"]
            )

        elif opt["model_name"] == "MBDCNN":
            model = MBDCNN(
                in_channels=opt["in_channels"],
                seg_out_channels=opt["seg_classes"] if opt["seg_classes"] is not None else 2,
                cls_out_channels=opt["cls_classes"] if opt["cls_classes"] is not None else 7
            )
        

        # proposed multitask model
        elif opt["model_name"] == "LiSANetMT":
            model = LiSANetMT(in_channels=opt["in_channels"], seg_out_channels=opt["seg_classes"], cls_out_channels=opt["cls_classes"] if opt["cls_classes"] is not None else 0, dim=opt["dimension"], scaling_version=opt["scaling_version"], segmentation=True, classification=True)


        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize model")

    # initialize model and weights
    model = model.to(opt["device"])
    utils.init_weights(model, init_type="kaiming")

    # initialize optimizer
    if opt["optimizer_name"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=opt["learning_rate"], momentum=opt["momentum"],
                              weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"],
                                  momentum=opt["momentum"])

    elif opt["optimizer_name"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "AdamW":
        # if multitask model, use different learning rates for different parts of the model
        if opt.get("task") == "multitask" and isinstance(model, LiSANetMT):
            optimizer = optim.AdamW([
                {"params": model.out_conv.parameters(), "lr": opt["lr_seg"]},
                {"params": model.classifier_fc.parameters(), "lr": opt["lr_cls"]},
                {"params": [
                    p for name, p in model.named_parameters()
                    if "out_conv" not in name and "classifier_fc" not in name
                ], "lr": opt["learning_rate"]}
            ], weight_decay=opt["weight_decay"])
        # else, use the same learning rate for all parts of the model
        else:
            if opt["optimizer_name"] == "AdamW":
                optimizer = optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    else:
        raise RuntimeError(f"No {opt['optimizer_name']} optimizer available")

    # initialize lr_scheduler
    if opt["lr_scheduler_name"] == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "MultiStepLR":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt["milestones"], gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt["T_max"])

    elif opt["lr_scheduler_name"] == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt["T_0"],
                                                                      T_mult=opt["T_mult"])

    elif opt["lr_scheduler_name"] == "OneCycleLR":
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt["learning_rate"],
                                                     steps_per_epoch=opt["steps_per_epoch"], epochs=opt["end_epoch"], cycle_momentum=False)

    elif opt["lr_scheduler_name"] == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=opt["mode"], factor=opt["factor"],
                                                            patience=opt["patience"])
    else:
        raise RuntimeError(f"No {opt['lr_scheduler_name']} lr_scheduler available")

    return model, optimizer, lr_scheduler


def get_model(opt):
    # initialize model
    if opt["dataset_name"] == "ISIC-2018":

        # segmentation only models
        if opt["model_name"] == "UNet":
            model = UNet(n_channels=opt["in_channels"], n_classes=opt["seg_classes"])

        elif opt["model_name"] == "EGEUNet":
            model = EGEUNet(input_channels=opt["in_channels"], num_classes=opt["seg_classes"])

        elif opt["model_name"] == "PMFSNet":
            model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["seg_classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"])

        elif opt["model_name"] == "LiSANet":
            model = LiSANet(in_channels=opt["in_channels"], out_channels=opt["seg_classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"])


        # classification only models
        elif opt["model_name"] == "ResNet50":
            model = ResNet50(num_classes=opt["cls_classes"], pretrained=True)

        elif opt["model_name"] == "DenseNet121":
            model = DenseNet121(num_classes=opt["cls_classes"], pretrained=True)

        elif opt["model_name"] == "EfficientNetV2":
            model = EfficientNetV2(num_classes=opt["cls_classes"], pretrained=True)

        elif opt["model_name"] in ["MobileNetV3", "mobilenetv3_large_100"]:
            model = MobileNetV3(num_classes=opt["cls_classes"], pretrained=True)


        # multitask models
        elif opt["model_name"] == "BreastCancerMT":
            model = BreastCancerMT(
                in_channels=opt["in_channels"],
                seg_out_channels=opt["seg_classes"],
                cls_out_channels=opt["cls_classes"]
            )

        elif opt["model_name"] == "MBDCNN":
            model = MBDCNN(
                in_channels=opt["in_channels"],
                seg_out_channels=opt["seg_classes"] if opt["seg_classes"] is not None else 2,
                cls_out_channels=opt["cls_classes"] if opt["cls_classes"] is not None else 7
            )

        # proposed multitask model
        elif opt["model_name"] == "LiSANetMT":
            model = LiSANetMT(in_channels=opt["in_channels"], seg_out_channels=opt["seg_classes"], cls_out_channels=opt["cls_classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"], segmentation=True, classification=True)

    
        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize model")

    model = model.to(opt["device"])

    return model
