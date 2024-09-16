from typing import Dict, Optional, Tuple,List
from loguru import logger
import torch
import torcheval
from tqdm import tqdm
import mlflow

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        metrics: List[torcheval.metrics.Metric],
        loss_fn: torch.nn._Loss,
        optimizer: torch.optim.Optimizer,
        device: Optional[str] = None,        
    ) -> None:
        
        self.model = model
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device    

        #TODO scheduler
        
    def fit(self, n_epochs, trainDataloader: torch.nn.DataLoader, validDataloader: torch.nn.DataLoader) -> None:
        for epoch in tqdm(range(n_epochs), colour="#1e4706"):
            train_loss = self.train(trainDataloader)

            for metric in self.metrics:
                metric.reset()
                
            test_loss = self.evaluate(validDataloader)
                      
            mlflow.log_metric("loss/train_epoch", train_loss, step=epoch)
            mlflow.log_metric("loss/test_epoch", test_loss, step=epoch)

            for metric in self.metrics:
                mlflow.log_metric(f"metric/{metric.__class__.__name__}", metric.compute(), step=epoch)                
           
            lr = [param_group["lr"] for param_group in self.optimizer.param_groups][0]
            mlflow.log_metric("learning_rate", lr, step=epoch)            

            metric_results = {metric.__class__.__name__: metric.compute() for metric in self.metrics}


            logger.info(
                f"Epoch {epoch} train {train_loss:.4f} test {test_loss:.4f} metric {metric_results}"  # noqa E501
            )

            #TODO early stopping
            
    def train(self, dataloader: torch.nn.DataLoader) -> float:
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
    
    def evaluate(self, dataloader: torch.nn.DataLoader) -> Tuple[Dict[str, float], float]:
        self.model.eval()
        valid_steps = len(dataloader)
        valid_loss: float = 0.0
        

        
        for _ in range(len(dataloader)):
            x, y = next(iter(dataloader))
            if self.device:
                x, y = x.to(self.device), y.to(self.device)
            yhat = self.model(x)
            valid_loss += self.loss_fn(yhat, y).cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            yhat = yhat.cpu().detach().numpy()
            for metric in self.metrics:
               metric.update(yhat,y)
                

        valid_loss /= valid_steps


        #TODO scheduler

        return valid_loss    


