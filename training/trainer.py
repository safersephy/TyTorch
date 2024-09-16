from typing import Dict, Optional, Tuple,List
from loguru import logger
import torch
from torcheval.metrics.metric import Metric
from torch.utils.data import DataLoader

from tqdm import tqdm
import mlflow


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
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

    def __call__(self, current_value: float) -> None:
        if self.best_value is None:
            self.best_value = current_value
        elif self._is_improvement(current_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, current_value: float) -> bool:
        if self.mode == 'min':
            return current_value < self.best_value - self.min_delta
        elif self.mode == 'max':
            return current_value > self.best_value + self.min_delta
        else:
            raise ValueError("Invalid mode. Choose 'min' or 'max'.")

    def reset(self) -> None:
        self.best_value = None
        self.counter = 0
        self.early_stop = False




class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        metrics: List[Metric[torch.Tensor]],
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[str] = None,
        early_stopping:Optional[EarlyStopping] = None        
    ) -> None:
        
        self.model = model
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.early_stopping = early_stopping    

        #TODO scheduler
        
    def fit(self, n_epochs, trainDataloader: DataLoader, validDataloader: DataLoader) -> None:
        for epoch in tqdm(range(n_epochs), colour="#1e4706"):
            train_loss = self.train(trainDataloader)

            for metric in self.metrics:
                metric.reset()
                
            val_loss = self.evaluate(validDataloader)
                      
            mlflow.log_metric("loss/train_epoch", train_loss, step=epoch)
            mlflow.log_metric("loss/test_epoch", val_loss, step=epoch)

            for metric in self.metrics:
                mlflow.log_metric(f"metric/{metric.__class__.__name__}", metric.compute(), step=epoch)                
           
            lr = [param_group["lr"] for param_group in self.optimizer.param_groups][0]
            mlflow.log_metric("learning_rate", lr, step=epoch)            

            metric_results = {metric.__class__.__name__: metric.compute() for metric in self.metrics}


            logger.info(
                f"Epoch {epoch} train {train_loss:.4f} test {val_loss:.4f} metric {metric_results}"  # noqa E501
            )

            
            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch+1}"
                        )
                    break
            
    def train(self, dataloader: DataLoader) -> float:
        self.model.train()
        train_loss: float = 0.0
        train_steps = len(dataloader)
        for _ in tqdm(range(train_steps), colour="#1e4706"):
            
            x, y = next(iter(dataloader))
            
            if self.device:
                x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.cpu().detach().numpy()
        train_loss /= train_steps

        return train_loss        
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, float], float]:
        self.model.eval()
        valid_steps = len(dataloader)
        valid_loss: float = 0.0
        

        
        for _ in range(len(dataloader)):
            x, y = next(iter(dataloader))
            if self.device:
                x, y = x.to(self.device), y.to(self.device)
            yhat = self.model(x)
            valid_loss += self.loss_fn(yhat, y).cpu().detach().numpy()
            y = y
            yhat = yhat
            for metric in self.metrics:
               metric.update(yhat,y)
                

        valid_loss /= valid_steps


        #TODO scheduler

        return valid_loss    