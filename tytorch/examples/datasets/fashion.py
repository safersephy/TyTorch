from pathlib import Path
import random
import torch
import numpy as np
from tytorch.data import DatasetFactory, TyTorchDataset
from tytorch.data_utils import split_train_val_direct, iter_valid_paths,load_image,check_create_folder


class FashionDatasetFactory(DatasetFactory):
    def __init__(self, bronze_folder, bronze_filename, silver_folder):
        super().__init__(bronze_folder,bronze_filename)
        self.silver_folder

    def transform(self, silver_filename, val_fraction):
        self.silver_filename = silver_filename
        data = torch.load(self.bronze_path, weights_only=False)  # type: ignore

        split_data = split_train_val_direct(data, val_fraction)

        self.silver_path = self.silver_folder / silver_filename
        torch.save(split_data, self.silver_path)

        return self.silver_path

    def load(
        self,
    ) -> tuple[
        TyTorchDataset,
        TyTorchDataset,
        TyTorchDataset,
    ]:
        data = torch.load(self.silver_path, weights_only=False)  # type: ignore

        train_data = data["traindata"]
        train_labels = data["trainlabels"]
        valid_data = data["validdata"]
        valid_labels = data["validlabels"]
        test_data = data["testdata"]
        test_labels = data["testlabels"]

        train_dataset = TyTorchDataset(train_data, train_labels)
        valid_dataset = TyTorchDataset(valid_data, valid_labels)
        test_dataset = TyTorchDataset(test_data, test_labels)

        return train_dataset, valid_dataset, test_dataset

    def create_datasets(self):
        self.extract(
            source_url="https://github.com/raoulg/data_assets/raw/main/fashionmnist.pt",
            bronze_filename="fashionmnist.pt",
            unzip=False,
        )

        self.transform(silver_filename="fashionmnist.pt", val_fraction=0.2)

        return self.load()


# train_dataset, valid_dataset, test_dataset = FashionDatasetFactory(
#     Path("./tytorch/examples/data/bronze"), Path("./tytorch/examples/data/silver")
# ).create_datasets()



