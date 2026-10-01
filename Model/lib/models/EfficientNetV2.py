import timm

def EfficientNetV2(num_classes, pretrained=False):
    model = timm.create_model(
        "tf_efficientnetv2_m",
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model
