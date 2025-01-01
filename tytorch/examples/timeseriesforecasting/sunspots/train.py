import warnings

import mlflow
import mlflow.pytorch
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tytorch.datapipeline.base import DataPipeline
from tytorch.examples.models.rnn import packed_lstm, lstm
from tytorch.datapipeline.global_transform_strategies import (
    SequenceTensorSplitStrategy,
    TensorNormalizeStrategy,
    TensorWindowStrategy
)
from tytorch.datapipeline.loader_strategies import (
    SunspotTensorLoaderStrategy,
)
from tytorch.trainer import EarlyStopping, Trainer
from tytorch.utils.mlflow import set_mlflow_experiment
from tytorch.metrics import MASE

warnings.simplefilter("ignore", UserWarning)

params = {
    "model_class": lstm,
    "batch_size": 32,
    "n_epochs": 50,
    "device": "cpu",
    "input_size": 1,
    "output_size": 3,
    "lr": 1e-3,
    "dropout": 0.5,
    "batch_first": True,
    "bidirectional": False,
    "hidden_size": 8,
    "num_layers": 1
}

source_url = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt"
bronze_folder = "./tytorch/examples/data/bronze"
bronze_filename = "sunspots.txt"
unzip = False
window_size=23
horizon=3
step=1


extract_loader = SunspotTensorLoaderStrategy(
    source_url=source_url,
    bronze_folder=bronze_folder,
    bronze_filename=bronze_filename,
    unzip=unzip,
)

global_strategies = [
    SequenceTensorSplitStrategy(
        test_frac=0.0, 
        valid_frac=0.2,
        shuffle_all=False,
        window_size=window_size,
        horizon=horizon,
        step=step),
    TensorNormalizeStrategy(),
    TensorWindowStrategy(window_size=window_size,horizon=horizon)
    ]

data_pipeline = DataPipeline(
    load_strategy=extract_loader, global_transform_strategies=global_strategies
)

train_dataset, val_dataset, _ = data_pipeline.create_datasets()


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32
)


model = params["model_class"](params)

optimizer = Adam(model.parameters(), lr=params["lr"])

trainer = Trainer(
    model=model,
    loss_fn=MSELoss(),
    metrics=[MASE(train_loader,3)],
    optimizer=optimizer,
    early_stopping=EarlyStopping(20, 0.01, mode="min"),
    device="cpu",
    lrscheduler=ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=5),
)

set_mlflow_experiment("train")
with mlflow.start_run():
    mlflow.log_params(params)
    trainer.fit(params["n_epochs"], train_loader, val_loader)

    mlflow.pytorch.log_model(model, artifact_path="logged_models/model")
mlflow.end_run()
