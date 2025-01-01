import warnings

import mlflow
import mlflow.pytorch
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torcheval.metrics import MulticlassAccuracy

from tytorch.datapipeline.base import DataPipeline
from tytorch.examples.models.rnn import packed_lstm, lstm
from tytorch.datapipeline.global_transform_strategies import SequenceTensorSplitStrategy
from tytorch.datapipeline.loader_strategies import GesturesTensorLoaderStrategy
from tytorch.trainer import EarlyStopping, Trainer
from tytorch.utils.data import pad_collate_packed,pad_collate
from tytorch.utils.mlflow import set_mlflow_experiment

warnings.simplefilter("ignore", UserWarning)


params = {
    "model_class": packed_lstm,
    "batch_size": 32,
    "n_epochs": 50,
    "lr": 1e-3,
}


params = {
    "model_class": packed_lstm,
    "batch_size": 32,
    "n_epochs": 50,
    "device": "cpu",
    "input_size": 3,
    "output_size": 20,
    "lr": 1e-3,
    "dropout": 0.5,
    "batch_first": True,
    "bidirectional": True,
    "hidden_size": 64,
    "num_layers": 5
}



source_url = "https://github.com/raoulg/gestures/raw/main/gestures-dataset.zip"
bronze_folder = "./tytorch/examples/data/bronze"
bronze_filename = "gestures.zip"
unzip = True

extract_loader = GesturesTensorLoaderStrategy(
    source_url=source_url,
    bronze_folder=bronze_folder,
    bronze_filename=bronze_filename,
    unzip=unzip,
)

global_strategies = [SequenceTensorSplitStrategy(test_frac=0.0, valid_frac=0.2)]

data_pipeline = DataPipeline(
    load_strategy=extract_loader, global_transform_strategies=global_strategies
)

train_dataset, val_dataset, test_dataset = data_pipeline.create_datasets()

train_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=32, collate_fn=pad_collate_packed
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, collate_fn=pad_collate_packed
)

model = params["model_class"](params)
optimizer = Adam(model.parameters(), lr=params["lr"])

trainer = Trainer(
    model=model,
    loss_fn=CrossEntropyLoss(),
    metrics=[MulticlassAccuracy()],
    optimizer=optimizer,
    early_stopping=EarlyStopping(5, 0.01, mode="min"),
    device="cpu",
    lrscheduler=ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=5),
)

set_mlflow_experiment("train")
with mlflow.start_run():
    mlflow.log_params(params)
    trainer.fit(params["n_epochs"], train_loader, val_loader)

    mlflow.pytorch.log_model(model, artifact_path="logged_models/model")
mlflow.end_run()
