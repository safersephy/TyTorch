from typing import Tuple

from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset

from tytorch.data_utils import get_file


class MappingDatasetFactory:
    def __init__(self):
        pass

    def extract(self, source_url, bronze_folder, bronze_filename, unzip):
        self.source_url = source_url
        self.bronze_folder = bronze_folder
        self.bronze_filename = bronze_filename
        self.unzip = unzip
        self.overwrite = False
        if not self.bronze_folder.exists():
            self.bronze_folder.mkdir(parents=True)

        if (self.bronze_folder / self.bronze_filename).exists():
            logger.info(
                f"File {self.bronze_filename} already exists at {self.bronze_folder}"
            )
            if not self.redownload:
                return None

        self.filepath = get_file(
            self.bronze_folder,
            self.bronze_filename,
            url=str(self.source_url),
            unzip=self.unzip,
            overwrite=self.overwrite,
        )

    def transform(self, silver_folder, silver_filename):
        self.silver_folder = silver_folder
        self.silver_filename = silver_filename

        # TODO output should be one dataframe saved to disc in a fixed template.

    def load(self, test_ratio, valid_ratio):
        # TODO input should be fixed template file, output should be datasets
        pass
        # train = TyTorchMappingDataset(training_data, training_labels)
        # valid = TyTorchMappingDataset(valid_data, valid_labels)
        # test = TyTorchMappingDataset(test_data, test_labels)
        # return train, valid, test


class TyTorchMappingDataset(Dataset):
    def __init__(self, data: "Tensor", labels: "Tensor"):
        super.__init__(Dataset)
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple:
        x = self.data[idx]
        y = self.labels[idx]
        return (x, y)

    def __repr__(self) -> str:
        return f"TyTorchMappingDataset (len {len(self)})"
