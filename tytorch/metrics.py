import torch
from torcheval.metrics import Metric
from torch import Tensor
from typing import Optional
import numpy as np

class MASE(Metric):
    def __init__(self, train:  torch.utils.data.DataLoader, horizon: int) -> None:
        """
        Initialize the MASE metric with training data for scaling.
        Args:
            train (BaseDatastreamer): Training data used to compute the scaling factor.
            horizon (int): Forecasting horizon.
        """
        super().__init__()
        self.horizon = horizon
        self.scale = self.naivenorm(train, horizon)
        self.reset()

    def naivenorm(self, train:  torch.utils.data.DataLoader, horizon: int) -> Tensor:
        """
        Compute the scaling factor using naive predictions on the training data.
        """
        elist = []
 
        for _ in range(len(train)):
            x, y = next(iter(train))
            yhat = self.naivepredict(x, horizon)
            e = self.mae(y.numpy(), yhat.numpy())
            elist.append(e)
        return torch.mean(torch.tensor(elist))

    def naivepredict(self, x: Tensor, horizon: int) -> Tensor:
        """
        Generate naive predictions using the last available values.
        """
        assert horizon > 0
        return x[..., -horizon:, :].squeeze(-1)

    def mae(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """
        Compute Mean Absolute Error (MAE).
        """
        return np.mean(np.abs(y - yhat))

    def update(self, y: Tensor, yhat: Tensor) -> None:
        """
        Update internal state with new predictions and ground truths.
        """
        mae_val = self.mae(y.detach().numpy(), yhat.detach().numpy())
        self.total_error += mae_val
        self.count += 1

    def compute(self) -> float:
        """
        Compute the final MASE value.
        """
        return (self.total_error / self.count) / self.scale

    def reset(self) -> None:
        """
        Reset internal state.
        """
        self.total_error = 0.0
        self.count = 0

    def __repr__(self) -> str:
        return f"MASE(scale={self.scale:.3f})"
    
    def merge_state(self, metrics) -> None:
        """
        Merge states from multiple metrics for distributed computing.
        
        Args:
            metrics: List of MASE metrics to merge from
        """
        for metric in metrics:
            self.sum_absolute_errors += metric.sum_absolute_errors
            self.sum_scale += metric.sum_scale
            self.total_samples += metric.total_samples