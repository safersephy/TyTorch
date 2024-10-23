import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import torch
from loguru import logger
from ray import train
from ray.train import Checkpoint
from torch.utils.data import DataLoader
from torcheval.metrics.metric import Metric
from torchinfo import summary
from tqdm import tqdm

from tytorch.utils.trainer_utils import step_requires_metric


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        save: bool = True,
        mode: str = "min",
    ) -> None:
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            mode (str): 'min' for minimizing loss, 'max' for maximizing metrics like accuracy.
        """
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.mode: str = mode
        self.best_value: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.save = save
        folder = Path("./earlystopping")
        self.path = folder / "checkpoint.pt"

        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, current_value: float, model: torch.nn.Module) -> None:
        if self.best_value is None:
            self.best_value = current_value
            self.save_checkpoint(current_value, model)
        elif self._is_improvement(current_value):
            self.best_value = current_value
            self.save_checkpoint(current_value, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, current_value: float) -> bool:
        if self.mode == "min":
            return current_value < self.best_value - self.min_delta  # type: ignore
        elif self.mode == "max":
            return current_value > self.best_value + self.min_delta  # type: ignore
        else:
            raise ValueError("Invalid mode. Choose 'min' or 'max'.")

    def reset(self) -> None:
        self.best_value = None
        self.counter = 0
        self.early_stop = False

    def save_checkpoint(self, current_value: float, model: torch.nn.Module) -> None:
        """Saves model when validation loss decrease."""
        if self.save:
            logger.info(
                f"Validation loss ({self.best_value:.4f} --> {current_value:.4f})."
                f"Saving {self.path} ..."
            )
            torch.save(model, self.path)

    def get_best(self) -> torch.nn.Module:
        return torch.load(self.path)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        metrics: List[Metric[torch.Tensor]],
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        early_stopping: Optional[EarlyStopping] = None,
        lrscheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        quiet: bool = False,
        train_steps: Optional[int] = None,
        valid_steps: Optional[int] = None,
    ) -> None:
        self.model = model
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.early_stopping = early_stopping
        self.lrscheduler = lrscheduler
        self.quiet = quiet
        self.train_steps = train_steps
        self.valid_steps = valid_steps

        if self.lrscheduler:
            self._lrscheduler_metric_step = step_requires_metric(self.lrscheduler)

    def fit(
        self,
        n_epochs: int,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
    ) -> None:
        if not self.quiet:
            summary(
                self.model, input_size=tuple((next(iter(train_dataloader))[0]).shape)
            )
        self.model.to(self.device)
        for epoch in tqdm(range(n_epochs), colour="#1e4706", disable=self.quiet):
            train_loss = self.train(train_dataloader)

            for metric in self.metrics:
                metric.reset()

            val_loss = self.evaluate(valid_dataloader)

            mlflow.log_metric("loss/train_epoch", train_loss, step=epoch)
            mlflow.log_metric("loss/val_epoch", val_loss, step=epoch)

            for metric in self.metrics:
                mlflow.log_metric(
                    f"metric/{metric.__class__.__name__}", metric.compute(), step=epoch
                )

            lr = [param_group["lr"] for param_group in self.optimizer.param_groups][0]
            mlflow.log_metric("learning_rate", lr, step=epoch)

            metric_results = {
                metric.__class__.__name__: metric.compute() for metric in self.metrics
            }

            logger.info(
                f"Epoch {epoch} train {train_loss:.4f} val {val_loss:.4f} metric {metric_results}"  # noqa E501
            )

            if self.lrscheduler:
                if self._lrscheduler_metric_step:
                    self.lrscheduler.step(val_loss)
                else:
                    self.lrscheduler.step()

            if self.early_stopping:
                self.early_stopping(val_loss, self.model)  # type: ignore
                if self.early_stopping.early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    logger.info("retrieving best model.")
                    self.model = self.early_stopping.get_best()
                    break

    def train(self, dataloader: DataLoader) -> float:
        self.model.train()
        train_loss: float = 0.0
        if not self.train_steps:
            self.train_steps = len(dataloader)
            
        for _ in tqdm(range(self.train_steps), colour="#1e4706", disable=self.quiet):
            x, y = next(iter(dataloader))
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.cpu().detach().numpy()
        train_loss /= self.train_steps

        return train_loss

    def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, float], float]:
        self.model.eval()
        if not self.valid_steps:
            self.valid_steps = len(dataloader)
        valid_loss: float = 0.0

        for _ in tqdm(range(self.valid_steps), colour="#1e4706", disable=self.quiet):
            x, y = next(iter(dataloader))
            x, y = x.to(self.device), y.to(self.device)
            yhat = self.model(x)
            valid_loss += self.loss_fn(yhat, y).cpu().detach().numpy()
            y = y
            yhat = yhat
            for metric in self.metrics:
                metric.update(yhat, y)

        valid_loss /= self.valid_steps

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            torch.save(self.model.state_dict(), Path(temp_checkpoint_dir) / "model.pth")
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            # Send the current training result back to Tune

            metric_results = {
                metric.__class__.__name__: float(metric.compute())
                for metric in self.metrics
            }

            train.report(
                {
                    **{
                        "valid_loss": valid_loss,
                    },
                    **metric_results,
                },
                checkpoint=checkpoint,
            )

        return valid_loss  # type: ignore
