import argparse
import tensorflow as tf
print(tf.__version__)

from wandb_addons.dataset import load_dataset

AUTOTUNE = tf.data.AUTOTUNE


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


    def get_dataloader(self, name: str):
        assert name in ["train", "val", "test"], "name must be one of train, val, test"
        dataloader = self._load_dataset()[name]

        if name=="train":
            dataloader = dataloader.shuffle(self.args.shuffle_buffer_size)

        dataloader = dataloader.map(self._preprocess_data, num_parallel_calls=AUTOTUNE)
        dataloader = dataloader.batch(self.args.batch_size)

        # TODO: add augmentation policies

        dataloader = dataloader.prefetch(AUTOTUNE)

        return dataloader
