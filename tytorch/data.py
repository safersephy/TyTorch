import abc

from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset

from tytorch.data_utils import check_create_folder, get_file


class TyTorchDataset(Dataset):
    def __init__(self, data: "Tensor", labels: "Tensor"):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __repr__(self) -> str:
        return f"TyTorchDataset (len {len(self)})"




class DatasetFactory(abc.ABC):
    def __init__(
        self, 
        bronze_folder:str = None, 
        bronze_filename:str = None,
        ):
        self.bronze_folder = bronze_folder
        self.bronze_filename = bronze_filename

    def extract(self, source_url, bronze_filename, unzip):
        self.source_url = source_url
        self.bronze_filename = bronze_filename
        self.unzip = unzip
        self.overwrite = False
        if not self.bronze_folder.exists():
            self.bronze_folder.mkdir(parents=True)

        if (self.bronze_folder / self.bronze_filename).exists():
            logger.info(
                f"File {self.bronze_filename} already exists at {self.bronze_folder}"
            )
            if not self.overwrite:
                self.bronze_path = self.bronze_folder / self.bronze_filename
                return None

        self.bronze_path = get_file(
            self.bronze_folder,
            self.bronze_filename,
            url=str(self.source_url),
            unzip=self.unzip,
            overwrite=self.overwrite,
        )

    @abc.abstractmethod
    def transform(self, silver_filename: str) -> None:
        """
        Transform the extracted data and save it to the silver_folder.
        This method must be implemented by the concrete subclass.
        """
        pass

    @abc.abstractmethod
    def load(self) -> tuple:
        """
        Load the transformed datasets.
        This method must be implemented by the concrete subclass.
        """
        pass

    def create_datasets(self):
        self.extract(self.source_url, self.bronze_filename, self.unzip)
        self.transform(self.silver_filename)
        return self.load()
