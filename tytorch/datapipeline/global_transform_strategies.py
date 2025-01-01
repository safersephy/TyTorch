from typing import Any, Optional, List

import torch

from tytorch.datapipeline.base import GlobalTransformStrategy
from tytorch.utils.data import split_data,ensure_3d_tensor


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


class SequenceTensorSplitStrategy(GlobalTransformStrategy):
    def __init__(
        self, 
        test_frac: float = 0.2, 
        valid_frac: float = 0.1, 
        shuffle_all: bool = True,
        window_size: int = None,
        horizon: int = None,
        step: int = None
    ) -> None:
        """
        Args:
            test_frac: Proportion of the dataset to include in the test split.
            valid_frac: Proportion of the training data to include in the validation split.
        """
        self.test_frac = test_frac
        self.valid_frac = valid_frac
        self.shuffle_all = shuffle_all
        self.window_size = window_size
        self.horizon = horizon
        self.step = step

    def apply(self, pipeline: Any) -> None:
        """
        Applies dataset splitting for both tensor and list inputs, supporting variable-length sequences.
        Handles both uniform-length tensors and variable-length sequences stored in lists.
        
        Args:
            pipeline: Pipeline object containing data and labels
        """
        # Determine if we're working with tensors or lists

        num_all_samples = len(pipeline.data)
        
        # Generate indices for the full dataset
        if self.shuffle_all:
            all_indices = torch.randperm(num_all_samples)
        else:
            all_indices = torch.arange(num_all_samples)
        
        # Calculate split sizes
        if all(var is not None for var in (self.window_size, self.horizon, self.step)):
            total_windows = (len(pipeline.data) - self.window_size - self.horizon) // self.step + 1
            train_val_windows = int(total_windows * (1 - self.test_frac))
            train_val_size = train_val_windows * self.step + self.window_size  
            train_val_indices = all_indices[:train_val_size]
            test_indices = all_indices[train_val_size:] if len(all_indices[train_val_size:]) >= (self.window_size + self.horizon) else None
                
        elif all(var is None for var in (self.window_size, self.horizon, self.step)):
            test_size = int(num_all_samples * self.test_frac)
            train_val_size = num_all_samples - test_size
            train_val_indices = all_indices[:train_val_size]
            test_indices = all_indices[train_val_size:] if len(all_indices[train_val_size:]) >= 0 else None
            
        else:
            raise ValueError("Either pass window, horizon, step or pass none")
   
        # Perform initial train/test split
        traintest_data_split = split_data(pipeline.data, train_val_indices)
        traintest_labels_split = split_data(pipeline.labels, train_val_indices) if pipeline.labels is not None else None
        
        num_train_samples = len(traintest_data_split)
        split_indices = torch.arange(num_train_samples)       
        
        # Calculate validation split  
        if all(var is not None for var in (self.window_size, self.horizon, self.step)):       
            total_windows = (len(traintest_data_split) - self.window_size - self.horizon) // self.step + 1           
            train_windows = int(total_windows * (1 - self.valid_frac))           
            train_size = train_windows * self.step + self.window_size  
            # Split indices for train/val
            train_indices = split_indices[:train_size] if len(split_indices[:train_size]) >= (self.window_size + self.horizon) else None
            val_indices = split_indices[train_size:] if len(split_indices[train_size:]) >= (self.window_size + self.horizon) else None     
            
        elif all(var is None for var in (self.window_size, self.horizon, self.step)):
            val_size = int(num_train_samples * self.valid_frac)
            train_size = num_train_samples - val_size
            # Split indices for train/val
            train_indices = split_indices[:train_size] if len(split_indices[:train_size]) > 0 else None
            val_indices = split_indices[train_size:] if len(split_indices[train_size:]) > 0 else None
        else:
            raise ValueError("Either pass window, horizon, step or pass none")
        
        # Store the splits in the pipeline
        if train_indices is not None:
            pipeline.train_data = split_data(traintest_data_split, train_indices)
            pipeline.train_labels = split_data(traintest_labels_split, train_indices) if pipeline.labels is not None else None
        else:
            pipeline.train_data = None
            pipeline.train_labels = None
            
        if val_indices is not None:
            pipeline.val_data = split_data(traintest_data_split, val_indices)
            pipeline.val_labels = split_data(traintest_labels_split, val_indices) if pipeline.labels is not None else None
        else:
            pipeline.val_data = None
            pipeline.val_labels = None
        
        if test_indices is not None:
            pipeline.test_data = split_data(pipeline.data, test_indices)
            pipeline.test_labels = split_data(pipeline.labels, test_indices) if pipeline.labels is not None else None
        else:
            pipeline.test_data = None
            pipeline.test_labels = None
        
        return None


class TensorNormalizeStrategy(GlobalTransformStrategy):
    def __init__(
        self
    ) -> None:     
        pass
    def apply(self, pipeline: Any) -> Optional[torch.Tensor]:
        norm = max(pipeline.train_data)
        pipeline.train_data = pipeline.train_data / norm
        pipeline.val_data = pipeline.val_data / norm if pipeline.val_data is not None else None
        pipeline.test_data = pipeline.test_data / norm if pipeline.test_data is not None else None

        return None  # No need to return data, we modify the pipeline in place



class TensorWindowStrategy(GlobalTransformStrategy):
    def __init__(
        self, window_size: int, horizon: int, step: int = 1
    ) -> None:
        self.window_size = window_size
        self.horizon = horizon
        self.step = step

    def apply(self, pipeline: Any) -> Optional[torch.Tensor]:
        
        if pipeline.train_data is not None:
            num_train_data_windows = (len(pipeline.train_data) - self.window_size) // self.step + 1
            num_train_label_windows = (len(pipeline.train_data) - self.window_size - self.horizon) // self.step + 1
            num_train_windows = min(num_train_data_windows, num_train_label_windows)
                
            pipeline.train_labels = pipeline.train_data[self.window_size:].unfold(dimension=0, size=self.horizon, step=self.step)[:num_train_windows]   
            pipeline.train_data = pipeline.train_data.unfold(dimension=0, size=self.window_size, step=self.step)[:num_train_windows]
            pipeline.train_data = ensure_3d_tensor(pipeline.train_data)
        if pipeline.val_data is not None:
            
            num_val_data_windows = (len(pipeline.val_data) - self.window_size) // self.step + 1
            num_val_label_windows = (len(pipeline.val_data) - self.window_size - self.horizon) // self.step + 1
            num_val_windows = min(num_val_data_windows, num_val_label_windows)
                       
            pipeline.val_labels = pipeline.val_data[self.window_size:].unfold(dimension=0, size=self.horizon, step=self.step)[:num_val_windows]                 
            pipeline.val_data = pipeline.val_data.unfold(dimension=0, size=self.window_size, step=self.step)[:num_val_windows]  
            pipeline.val_data = ensure_3d_tensor(pipeline.val_data)               
        if pipeline.test_data is not None:

            num_test_data_windows = (len(pipeline.test_data) - self.window_size) // self.step + 1
            num_test_label_windows = (len(pipeline.test_data) - self.window_size - self.horizon) // self.step + 1
            num_test_windows = min(num_test_data_windows, num_test_label_windows)
                   
            pipeline.test_labels = pipeline.test_data[self.window_size:].unfold(dimension=0, size=self.horizon, step=self.step)[:num_test_windows]            
            pipeline.test_data = pipeline.test_data.unfold(dimension=0, size=self.window_size, step=self.step)[:num_test_windows]      
            pipeline.test_data = ensure_3d_tensor(pipeline.test_data)
        return None  # No need to return data, we modify the pipeline in place
