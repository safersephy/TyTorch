import warnings

import mlflow
import mlflow.pytorch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy

from tytorch.datapipeline import DataPipeline
from tytorch.examples.models.CustomResNet import ResNet
from tytorch.strategies.global_transform_strategies import ImageTensorSplitStrategy
from tytorch.strategies.item_transform_strategies import ImageTensorAugmentationStrategy
from tytorch.strategies.loader_strategies import ImageTensorLoaderStrategy
from tytorch.trainer import EarlyStopping, Trainer
from tytorch.utils.mlflow import get_training_config, set_mlflow_experiment

warnings.simplefilter("ignore", UserWarning)

params = get_training_config()
if params is None:
    # params = {
    #     "model_class": CNN,
    #     "batch_size": 32,
    #     "n_epochs": 100,
    #     "input_size": (32, 3, 224, 224),
    #     "output_size": 5,
    #     "lr": 1e-4,
    #     "dropout": 0.3,
    #     "conv_blocks": [
    #         {"num_conv_layers": 6, "initial_filters": 32,"growth_factor": 2, "pool": True, "residual": True},
    #         {"num_conv_layers": 4, "initial_filters": 256,"growth_factor": 1, "pool": False, "residual": False},
    #     ],
    #     "linear_blocks": [
    #         #{"out_features": 32, "dropout": 0.0},
    #         {"out_features": 16, "dropout": 0.0},
    #     ],
    # }

    params = {
        "model_class": ResNet,
        "batch_size": 32,
        "n_epochs": 100,
        "lr": 1e-4,
        "input_size": (32, 3, 224, 224),
    }


source_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
bronze_folder = "./tytorch/examples/data/bronze"
bronze_filename = "flowers.tgz"
unzip = True


extract_loader = ImageTensorLoaderStrategy(
    source_url=source_url,
    bronze_folder=bronze_folder,
    bronze_filename=bronze_filename,
    unzip=unzip,
)

global_strategies = [ImageTensorSplitStrategy(test_frac=0.2, valid_frac=0.2)]

data_pipeline = DataPipeline(
    load_strategy=extract_loader, global_transform_strategies=global_strategies
)

item_transform_strategies = [ImageTensorAugmentationStrategy()]


train_dataset, val_dataset, test_dataset = data_pipeline.create_datasets(
    item_transform_strategies
)

trainloader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
testloader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=True)

model = params["model_class"](img_channels=3, num_classes=5)
optimizer = Adam(model.parameters(), lr=params["lr"])

trainer = Trainer(
    model=model,
    loss_fn=CrossEntropyLoss(),
    metrics=[MulticlassAccuracy()],
    optimizer=optimizer,
    early_stopping=EarlyStopping(10, 0.01, mode="min"),
    device="mps",
    lrscheduler=ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3),
)

set_mlflow_experiment("train")
with mlflow.start_run():
    mlflow.log_params(params)
    trainer.fit(params["n_epochs"], trainloader, testloader)

    mlflow.pytorch.log_model(model, artifact_path="logged_models/model")
mlflow.end_run()
