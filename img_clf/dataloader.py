import argparse
import tensorflow as tf
print(tf.__version__)

from wandb_addons.dataset import load_dataset
from keras_cv.layers import preprocessing


AUTOTUNE = tf.data.AUTOTUNE


base_augmentations = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02),
    ],
    name="base_augmentation",
)

mixup = preprocessing.MixUp(alpha=0.8)


class GetDataloader:
    def __init__(
            self,
            args: argparse.Namespace,
            dataset_path: str = "ml-colabs/e2e-img-clf/cards_dataset:v1"
        ):
        self.args = args
        self.dataset_path = dataset_path

    def _load_dataset(self):
        datasets, dataset_builder_info = load_dataset(self.dataset_path)
        return datasets

    def _preprocess_data(self, example):
        image = example["image"]
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (self.args.img_height, self.args.img_width))

        label = example["label"]
        if self.args.one_hot:
            label = tf.one_hot(label, self.args.num_classes)

        return image, label

    def _apply_base_augmentations(self, images, labels):
        images = base_augmentations(images)
        return images, labels

    def get_dataloader(self, name: str):
        assert name in ["train", "val", "test"], "name must be one of train, val, test"
        dataloader = self._load_dataset()[name]

        if name=="train":
            dataloader = dataloader.shuffle(self.args.shuffle_buffer_size)

        dataloader = dataloader.map(self._preprocess_data, num_parallel_calls=AUTOTUNE)
        dataloader = dataloader.batch(self.args.batch_size)

        if name=="train":
            dataloader = (
                dataloader
                .map(self._apply_base_augmentations, num_parallel_calls=AUTOTUNE)
                .map(lambda images, labels: mixup({"images": images, "labels": labels}), num_parallel_calls=AUTOTUNE)
                .map(lambda x: (x["images"], x["labels"]), num_parallel_calls=AUTOTUNE)
            )

        dataloader = dataloader.prefetch(AUTOTUNE)

        return dataloader
