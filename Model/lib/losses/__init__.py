# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 16:56
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch

from .DiceLoss import DiceLoss
from .CrossEntropyLoss import CrossEntropyLoss

def get_loss_function(opt):
    loss_functions = {}
    device = opt.get("device", "cpu")
    
    if opt.get("segmentation", False):
        seg_classes = opt.get("seg_classes", 1)  
        loss_functions["segmentation"] = DiceLoss(
            classes=seg_classes,
            weight=None,
            sigmoid_normalization=opt.get("sigmoid_normalization", False),
            mode=opt.get("dice_loss_mode", "standard")
        )
    
    if opt.get("classification", False):
        # experiment: applying class weights to the classification loss
        #  - use the supplied weights when use_class_weight is True
        #  - otherwise, leave the loss unweighted
        #  *note: weights are read from the settings, not calculated here
        class_weight = None

        if opt.get("use_class_weight", False):
            class_weight = opt.get("class_weight", None)

            if class_weight is not None:
                class_weight = torch.FloatTensor(class_weight).to(device)

        loss_functions["classification"] = CrossEntropyLoss(weight=class_weight)

    if not loss_functions:
        raise RuntimeError(f"No {opt['loss_function_name']} is available")
    
    return loss_functions
