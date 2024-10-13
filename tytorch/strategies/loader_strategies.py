from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch

from tytorch.datapipeline import DataLoadingStrategy
from tytorch.utils.data import download_data, iter_valid_paths, load_image

DatasetDataType = torch.Tensor | np.ndarray | List | pd.DataFrame | pl.DataFrame | None
DatasetLabelType = torch.Tensor | np.ndarray | List | pd.DataFrame | None


class ImageTensorLoaderStrategy(DataLoadingStrategy):
    def __init__(
        self,
        source_url: str,
        bronze_folder: str,
        bronze_filename: str,
        unzip: bool = False,
        overwrite: bool = False,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.source_url = source_url
        self.bronze_folder = Path(bronze_folder)  # Keep this as Path for internal usage
        self.bronze_filename = bronze_filename
        self.unzip = unzip
        self.overwrite = overwrite
        self.image_size = image_size

    def set_dataset(self, dataset: Any) -> None:
        self.dataset = dataset

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert Path object to str before passing to download_data
        download_data(
            source_url=self.source_url,
            bronze_folder=str(self.bronze_folder),
            bronze_filename=self.bronze_filename,
            unzip=self.unzip,
            overwrite=self.overwrite,
        )

        paths_, class_names = iter_valid_paths(
            self.bronze_folder / "flower_photos", [".jpg", ".png"]
        )

        # Add type annotation for all_data
        all_data: Dict[str, List[torch.Tensor]] = {
            "data": [],
            "labels": [],
        }

        for path in paths_:
            img = load_image(path, self.image_size)
            x_ = np.transpose(img, (2, 0, 1))
            x = torch.tensor(x_ / 255.0).type(torch.float32)  # type: ignore
            y = torch.tensor(class_names.index(path.parent.name))
            all_data["data"].append(x)
            all_data["labels"].append(y)

        all_data["data"] = torch.stack(all_data["data"])  # Stack to create a tensor
        all_data["labels"] = torch.tensor(all_data["labels"])

        return all_data["data"], all_data["labels"]
