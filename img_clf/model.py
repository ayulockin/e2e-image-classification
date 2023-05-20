import argparse
import tensorflow as tf
from tensorflow.keras import layers, models


def get_convnext_model(model_name: str, weights="imagenet"):
    variant = model_name.split("-")[-1]
    if variant == "b":
        backbone = tf.keras.applications.convnext.ConvNeXtBase
    if variant == "s":
        backbone = tf.keras.applications.convnext.ConvNeXtSmall
    if variant == "t":
        backbone = tf.keras.applications.convnext.ConvNeXtTiny

    return backbone


def get_effnetv2_backbone(model_name: str, weights: str = "imagenet"):
    variant = model_name.split("-")[-1]
    if variant == "b2":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2B2
    if variant == "b0":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2B0
    if variant == "s":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2S

    return backbone


def get_backbone(args: argparse.Namespace):
    """Get backbone for the model."""
    if args.model_backbone == "vgg16":
        base_model = tf.keras.applications.VGG16
    elif args.model_backbone == "resnet50":
        base_model = tf.keras.applications.ResNet50
    elif "convnext" in args.model_backbone:
        base_model = get_convnext_model(args.model_backbone)
    elif "effnetv2" in args.model_backbone:
        base_model = get_effnetv2_backbone(args.model_backbone)
    else:
        raise NotImplementedError("Not implemented for this backbone.")

    return base_model


def get_model(args: argparse.Namespace):
    """Get an image classifier with a CNN based backbone."""
    # Stack layers
    inputs = layers.Input(
        shape=(
            args.img_height,
            args.img_width,
            args.img_channels,
        )
    )

    # Backbone
    base_model = get_backbone(args)
    backbone = base_model(include_top=False, weights="imagenet", input_tensor=inputs)

    if args.freeze_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True

    x = backbone.layers[-1].output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(args.dropout_rate)(x)
    outputs = layers.Dense(args.num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)
