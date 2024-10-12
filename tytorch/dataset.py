from typing import Any, List, Optional, Sequence, TypeAlias

import numpy as np
import pandas as pd
import polars as pl
import torch

# Define the data types as type aliases
DatasetDataType: TypeAlias = (
    torch.Tensor | np.ndarray | List | pd.DataFrame | pl.DataFrame
)
DatasetLabelType: TypeAlias = torch.Tensor | np.ndarray | List | pd.DataFrame


class TyTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Optional[DatasetDataType],
        labels: Optional[DatasetLabelType],
        item_transform_strategies: Optional[Sequence[Any]] = None,
    ) -> None:
        self.data: Optional[DatasetDataType] = data
        self.labels: Optional[DatasetLabelType] = (
            labels  # Labels are expected to be provided separately
        )
        self.item_transform_strategies: Sequence[Any] = item_transform_strategies or []

    def __len__(self) -> int:
        if self.data is None:
            return 0
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        if self.data is None or self.labels is None:
            raise ValueError("Data or labels are not properly initialized.")

        sample = self.data[idx]
        label = self.labels[idx]

        # Apply per-item transformations
        for strategy in self.item_transform_strategies:
            sample = strategy.apply(sample)

        return sample, label
