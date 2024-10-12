import torch
from torcheval.metrics.metric import Metric


class MASE(Metric):
    def __init__(self, seasonality: int = 1):
        super().__init__()
        self.seasonality = seasonality
        self.add_state("absolute_errors", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scale", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("Predictions and targets must have the same shape")
        absolute_error = torch.abs(preds - target)
        self.absolute_errors += torch.sum(absolute_error)
        scale = torch.mean(
            torch.abs(target[self.seasonality :] - target[: -self.seasonality])
        )
        if scale == 0:
            scale = torch.tensor(1.0)
        self.scale += scale
        self.count += target.numel()

    def compute(self) -> torch.Tensor:
        mase = (self.absolute_errors / self.count) / (self.scale / self.count)
        return mase

    def reset(self) -> None:
        self.absolute_errors = torch.tensor(0.0)
        self.scale = torch.tensor(0.0)
        self.count = torch.tensor(0)
