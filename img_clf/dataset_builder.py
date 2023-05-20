#
import os
import json
import pandas as pd
from glob import glob
from typing import Any, Mapping, Optional, Union

from etils import epath
import tensorflow_datasets as tfds

import wandb
from wandb_addons.dataset import WandbDatasetBuilder


df = pd.read_csv("data/cards.csv")
_CLASS_LABELS = df["labels"].unique().tolist()
_CLASS_LABELS = {label: i for i, label in enumerate(_CLASS_LABELS)}
json.dump(_CLASS_LABELS, open("data/labels.json", "w"))

_DESCRIPTION = "Cards Dataset"


class CardsDatasetBuilder(WandbDatasetBuilder):
    def __init__(
        self,
        *,
        name: str,
        dataset_path: str,
        features: tfds.features.FeatureConnector,
        upload_raw_dataset: bool = True,
        config: Union[None, str, tfds.core.BuilderConfig] = None,
        data_dir: Optional[epath.PathLike] = None,
        description: Optional[str] = None,
        release_notes: Optional[Mapping[str, str]] = None,
        homepage: Optional[str] = None,
        file_format: Optional[Union[str, tfds.core.FileFormat]] = None,
        disable_shuffling: Optional[bool] = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            dataset_path=dataset_path,
            features=features,
            upload_raw_dataset=upload_raw_dataset,
            config=config,
            description=description,
            data_dir=data_dir,
            release_notes=release_notes,
            homepage=homepage,
            file_format=file_format,
            disable_shuffling=disable_shuffling,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples(os.path.join(self.dataset_path, "train")),
            "val": self._generate_examples(os.path.join(self.dataset_path, "valid")),
            "test": self._generate_examples(os.path.join(self.dataset_path, "test")),
        }

    def _generate_examples(self, path):
        image_paths = glob(os.path.join(path, "*", "*.jpg"))
        for image_path in image_paths:
            label = _CLASS_LABELS[image_path.split("/")[-2]]
            yield image_path, {
                "image": image_path,
                "label": label,
            }


if __name__ == "__main__":
    wandb.init(project="e2e-img-clf", entity="ml-colabs")

    builder = CardsDatasetBuilder(
        name="cards_dataset",
        dataset_path="data/",
        features=tfds.features.FeaturesDict(
            {
                "image": tfds.features.Image(shape=(None, None, 3)),
                "label": tfds.features.ClassLabel(names=_CLASS_LABELS),
            }
        ),
        data_dir="data/",
        description=_DESCRIPTION,
    )

    builder.build_and_upload(
        create_visualizations=True, max_visualizations_per_split=100
    )
    # builder.download_and_prepare()
