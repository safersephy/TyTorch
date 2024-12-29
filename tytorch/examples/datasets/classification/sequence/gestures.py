import torch

from tytorch.datapipeline import DataPipeline
from tytorch.strategies.global_transform_strategies import (
    SequenceTensorSplitStrategy,
)
from tytorch.strategies.loader_strategies import (
    GesturesTensorLoaderStrategy,
)
from tytorch.utils.data import pad_collate

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

global_strategies = [SequenceTensorSplitStrategy(test_frac=0.2, valid_frac=0.2)]

data_pipeline = DataPipeline(
    load_strategy=extract_loader, global_transform_strategies=global_strategies
)

train_dataset, val_dataset, test_dataset = data_pipeline.create_datasets()

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, collate_fn=pad_collate
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, collate_fn=pad_collate
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, collate_fn=pad_collate
)

# Access batches from the train loader
for data, labels in train_loader:
    print(data.shape, labels.shape)
