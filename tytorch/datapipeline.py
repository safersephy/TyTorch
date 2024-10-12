from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, TypeAlias

import numpy as np
import pandas as pd
import polars as pl
import torch

from tytorch.dataset import TyTorchDataset

# Define type aliases for DatasetDataType and DatasetLabelType
DatasetDataType: TypeAlias = (
    torch.Tensor | np.ndarray | List | pd.DataFrame | pl.DataFrame | None
)
DatasetLabelType: TypeAlias = torch.Tensor | np.ndarray | List | pd.DataFrame | None


class DataLoadingStrategy(ABC):
    @abstractmethod
    def load_data(self) -> Tuple[Optional[DatasetDataType], Optional[DatasetLabelType]]:
        """
        Load the dataset. This method should return a data object or None if
        no data needs to be loaded at this step.
        If global transformations are to be applied, they should modify the data directly.

        Returns:
            data: The loaded data or None if the data is already available.
        """
        pass


class GlobalTransformStrategy(ABC):
    def set_pipeline(self, pipeline: Any) -> None:
        """
        Optional method to provide access to the DataPipeline instance, allowing
        the global transformation strategy to interact with the data and labels.
        """
        self.pipeline = pipeline

    @abstractmethod
    def apply(self, pipeline: Any) -> Optional[DatasetDataType]:
        """
        Apply a global transformation to the dataset.

        Args:
            pipeline: The DataPipeline instance which holds the data and labels.

        Returns:
            data: The transformed data or None if the transformation modifies the pipeline's data/labels in place.
        """
        pass


class ItemTransformStrategy(ABC):
    def set_pipeline(self, pipeline: Any) -> None:
        """
        Optional method to provide access to the DataPipeline instance, allowing
        the item transformation strategy to interact with the data or metadata if needed.
        """
        self.pipeline = pipeline

    @abstractmethod
    def apply(self, data: DatasetDataType) -> DatasetDataType:
        """
        Apply a transformation to an individual data sample.

        Args:
            data: The individual sample to be transformed.

        Returns:
            transformed_data: The transformed data sample.
        """
        pass


class DataPipeline:
    def __init__(
        self,
        load_strategy: DataLoadingStrategy,
        global_transform_strategies: Optional[Sequence[GlobalTransformStrategy]] = None,
    ) -> None:
        self.load_strategy = load_strategy
        self.global_transform_strategies: Sequence[GlobalTransformStrategy] = (
            global_transform_strategies or []
        )

        self.data: DatasetDataType = None
        self.labels: DatasetLabelType = None
        # Load data
        self.data, self.labels = self.load_strategy.load_data()

        # To store split data
        self.train_data: Optional[DatasetDataType] = None
        self.train_labels: Optional[DatasetLabelType] = None
        self.val_data: Optional[DatasetDataType] = None
        self.val_labels: Optional[DatasetLabelType] = None
        self.test_data: Optional[DatasetDataType] = None
        self.test_labels: Optional[DatasetLabelType] = None

        # Apply global transformations (including splitting)
        self.apply_global_transformations()

    def apply_global_transformations(self) -> None:
        """Apply global transformations to the entire dataset."""
        for strategy in self.global_transform_strategies:
            transformed_data = strategy.apply(self)
            if transformed_data is not None:
                self.data = transformed_data  # Update self.data only if the strategy returns new data

    def create_datasets(
        self,
        item_transform_strategies: Optional[Sequence[ItemTransformStrategy]] = None,
    ) -> Tuple[TyTorchDataset, TyTorchDataset, TyTorchDataset]:
        """
        Creates and returns three TyTorchDataset instances for train, validation, and test sets.

        Returns:
            train_dataset: TyTorchDataset for the training set.
            val_dataset: TyTorchDataset for the validation set.
            test_dataset: TyTorchDataset for the test set.
        """
        self.item_transform_strategies: Sequence[ItemTransformStrategy] = (
            item_transform_strategies or []
        )

        if self.train_data is None or self.val_data is None or self.test_data is None:
            raise ValueError(
                "Data splits have not been generated. Ensure the SplitStrategy has been applied."
            )

        # Create TyTorchDataset instances for each split
        train_dataset = TyTorchDataset(
            data=self.train_data,
            labels=self.train_labels,
            item_transform_strategies=self.item_transform_strategies,
        )
        val_dataset = TyTorchDataset(
            data=self.val_data,
            labels=self.val_labels,
            item_transform_strategies=self.item_transform_strategies,
        )
        test_dataset = TyTorchDataset(
            data=self.test_data,
            labels=self.test_labels,
            item_transform_strategies=self.item_transform_strategies,
        )

        return train_dataset, val_dataset, test_dataset
