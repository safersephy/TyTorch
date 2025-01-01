from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import TuneConfig
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy

from tytorch.datapipeline.base import DataPipeline
from tytorch.examples.models.rnn import packed_lstm, lstm
from tytorch.datapipeline.global_transform_strategies import SequenceTensorSplitStrategy
from tytorch.datapipeline.loader_strategies import GesturesTensorLoaderStrategy
from tytorch.trainer import Trainer
from tytorch.utils.mlflow import set_best_run_tag_and_log_model, set_mlflow_experiment
from tytorch.utils.data import pad_collate_packed,pad_collate

# initial params
tuningmetric = "valid_loss"
tuninggoal = "min"
n_trials = 60

params = {
    "model_class": packed_lstm,
    "batch_size": 32,
    "n_epochs": 10,
    "device": "cpu",
    "input_size": 3,
    "output_size": 20,
    "lr": 1e-3,
    "dropout": 0.5,
    "batch_first": True,
    "bidirectional": True,
    "hidden_size": tune.qrandint(16, 64, 16),
    "num_layers": tune.randint(1,6)
}

experiment_name = set_mlflow_experiment("tune")

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




def tune_func(config: dict) -> None:


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

    if callable(config["model_class"]):
        model = config["model_class"](config)
    else:
        raise TypeError("model_class should be a callable class.")

    optimizer = Adam(model.parameters(), lr=params["lr"])
    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        metrics=[MulticlassAccuracy()],
        optimizer=optimizer,
        device=config["device"],
        lrscheduler=ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=5),
        quiet=True,
    )

    mlflow.log_params(config)

    if not isinstance(params["n_epochs"], int):
        raise TypeError(
            f"Expected n_epochs to be an int, got {type(params['n_epochs'])} instead"
        )

    n_epochs = int(params.get("n_epochs", 10))  # type: ignore

    trainer.fit(n_epochs, train_loader, val_loader)


tuner = tune.Tuner(
    tune.with_resources(tune_func, {"cpu": 10}),
    param_space=params,
    tune_config=TuneConfig(
        mode=tuninggoal,
        #search_alg=TuneBOHB(),
        #scheduler=HyperBandForBOHB(),
        metric=tuningmetric,
        num_samples=n_trials,
        max_concurrent_trials=1,
    ),
    run_config=train.RunConfig(
        storage_path=Path("./ray_tuning_results").resolve(),
        name=experiment_name,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=mlflow.get_tracking_uri(),
                experiment_name=experiment_name,
                save_artifact=True,
            )
        ],
    ),
)
results = tuner.fit()

best_result = results.get_best_result(tuningmetric, tuninggoal)
model = params["model_class"](best_result.config)  # type: ignore
model.load_state_dict(torch.load(Path(best_result.checkpoint.path) / "model.pth"))
set_best_run_tag_and_log_model(experiment_name, model, tuningmetric, tuninggoal)
