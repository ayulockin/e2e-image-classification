import argparse
import tensorflow as tf
from tensorflow.keras import layers, models


def get_convnext_model(model_name: str, weights="imagenet"):
    variant = model_name.split("-")[-1]
    if variant == "b":
        backbone = tf.keras.applications.convnext.ConvNeXtBase(
            include_top=False,
            weights=weights
        )
    if variant == "s":
        backbone = tf.keras.applications.convnext.ConvNeXtSmall(
            include_top=False,
            weights=weights
        )
    if variant == "t":
        backbone = tf.keras.applications.convnext.ConvNeXtTiny(
            include_top=False,
            weights=weights
        )

    return backbone


def get_effnetv2_backbone(model_name: str, weights:str = "imagenet"):
    variant = model_name.split("-")[-1]
    if variant == "b2":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(
            include_top=False,
            weights=weights
        )
    if variant == "b0":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
            include_top=False,
            weights=weights
        )
    if variant == "s":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False,
            weights=weights
        )

    return backbone


def get_backbone(args: argparse.Namespace):
    """Get backbone for the model."""
    if args.model_backbone == "vgg16":
        base_model = tf.keras.applications.VGG16(include_top=False)
        base_model.trainable = True
    elif args.model_backbone == "resnet50":
        base_model = tf.keras.applications.ResNet50(include_top=False)
        base_model.trainable = True
    elif "convnext" in args.model_backbone:
        base_model = get_convnext_model(args.model_backbone)
        base_model.trainable = True
    elif "effnetv2" in args.model_backbone:
        base_model = get_effnetv2_backbone(args.model_backbone)
        base_model.trainable = True
    else:
        raise NotImplementedError("Not implemented for this backbone.")

    return base_model


def get_model(args: argparse.Namespace):
    """Get an image classifier with a CNN based backbone."""
    # Backbone
    base_model = get_backbone(args)

    # Stack layers
    inputs = layers.Input(
        shape=(
            args.img_height,
            args.img_width,
            args.img_channels,
        )
    )

    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(args.dropout_rate)(x)
    outputs = layers.Dense(args.num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)
