import torch
from torch import Tensor, nn
from tytorch.datapipeline import DataPipeline
from tytorch.strategies.global_transform_strategies import (
    SequenceTensorSplitStrategy,
    TensorNormalizeStrategy,
    TensorWindowStrategy
)
from tytorch.strategies.loader_strategies import (
    SunspotTensorLoaderStrategy,
)
from tytorch.utils.data import pad_collate

from tytorch.metrics import MASE

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

# Access batches from the train loader
for data, labels in train_loader:
    print(data.shape, labels.shape)




