from typing import Any, Optional

import torch

from tytorch.datapipeline import GlobalTransformStrategy


class ImageTensorSplitStrategy(GlobalTransformStrategy):
    def __init__(self, test_frac: float = 0.2, valid_frac: float = 0.1) -> None:
        """
        Args:
            test_frac: Proportion of the dataset to include in the test split.
            valid_frac: Proportion of the training data to include in the validation split.
        """
        self.test_frac = test_frac
        self.valid_frac = valid_frac

    def apply(self, pipeline: Any) -> Optional[torch.Tensor]:
        """Applies the dataset split and stores the resulting datasets in the pipeline."""
        if pipeline.labels is not None:
            num_all_samples = len(pipeline.data)
            all_indices = torch.randperm(num_all_samples)

            test_size = int(num_all_samples * self.test_frac)
            train_size = num_all_samples - test_size

            train_indices = all_indices[:train_size]
            test_indices = all_indices[train_size:]

            traintest_data_split = pipeline.data[train_indices]
            traintest_labels_split = pipeline.labels[train_indices]

            num_train_samples = len(traintest_data_split)
            split_indices = torch.randperm(num_train_samples)

            val_size = int(num_train_samples * self.valid_frac)
            train_size = num_train_samples - val_size

            train_indices = split_indices[:train_size]
            val_indices = split_indices[train_size:]

            # Store the splits in the pipeline
            pipeline.train_data = traintest_data_split[train_indices]
            pipeline.train_labels = traintest_labels_split[train_indices]
            pipeline.val_data = traintest_data_split[val_indices]
            pipeline.val_labels = traintest_labels_split[val_indices]
            pipeline.test_data = pipeline.data[test_indices]
            pipeline.test_labels = pipeline.labels[test_indices]
        else:
            raise ValueError("Cannot split dataset without explicit labels.")

        return None  # No need to return data, we modify the pipeline in place
