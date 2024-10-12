from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import TuneConfig
from ray.tune.search.hyperopt import HyperOptSearch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy

from tytorch.datapipeline import DataPipeline
from tytorch.examples.models.CustomCNN import CNN
from tytorch.strategies.global_transform_strategies import ImageTensorSplitStrategy
from tytorch.strategies.item_transform_strategies import ImageTensorAugmentationStrategy
from tytorch.strategies.loader_strategies import ImageTensorLoaderStrategy
from tytorch.trainer import Trainer
from tytorch.utils.mlflow import set_best_run_tag_and_log_model, set_mlflow_experiment

tuningmetric = "valid_loss"
tuninggoal = "min"
n_trials = 60

params = {
    "model_class": CNN,
    "batch_size": 32,
    "n_epochs": 4,
    "device": "mps",
    "input_size": (32, 3, 224, 224),
    "output_size": 5,
    "lr": 1e-4,
    "dropout": 0.3,
    "conv_blocks": [
        {
            "num_conv_layers": tune.randint(1, 3),
            "initial_filters": 32,
            "growth_factor": 2,
            "pool": True,
            "residual": True,
        },
        {
            "num_conv_layers": tune.randint(1, 3),
            "initial_filters": 256,
            "growth_factor": 1,
            "pool": False,
            "residual": False,
        },
    ],
    "linear_blocks": [
        {"out_features": 32, "dropout": 0.0},
        {"out_features": 16, "dropout": 0.0},
    ],
}

experiment_name = set_mlflow_experiment("tune")

source_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
bronze_folder = Path("./tytorch/examples/data/bronze").resolve()
bronze_filename = "flowers.tgz"
unzip = True

extract_loader = ImageTensorLoaderStrategy(
    source_url=source_url,
    bronze_folder=str(bronze_folder),  # Convert Path to str
    bronze_filename=bronze_filename,
    unzip=unzip,
    overwrite=False,
)

global_strategies = [ImageTensorSplitStrategy(test_frac=0.2, valid_frac=0.2)]

item_transform_strategies = [ImageTensorAugmentationStrategy()]


def tune_func(config: dict) -> None:
    for idx, block in enumerate(config["conv_blocks"]):
        config[f"conv_block_{idx}_num_conv_layers"] = block["num_conv_layers"]
        config[f"conv_block_{idx}_initial_filters"] = block["initial_filters"]
        config[f"conv_block_{idx}_growth_factor"] = block["growth_factor"]
        config[f"conv_block_{idx}_pool"] = block["pool"]
        config[f"conv_block_{idx}_residual"] = block["residual"]

    data_pipeline = DataPipeline(
        load_strategy=extract_loader, global_transform_strategies=global_strategies
    )

    train_dataset, val_dataset, _ = data_pipeline.create_datasets(
        item_transform_strategies
    )

    trainloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    testloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)

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

    trainer.fit(n_epochs, trainloader, testloader)


tuner = tune.Tuner(
    tune.with_resources(tune_func, {"cpu": 10}),
    param_space=params,
    tune_config=TuneConfig(
        mode=tuninggoal,
        search_alg=HyperOptSearch(),
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
