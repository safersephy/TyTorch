import torch

from tytorch.datapipeline import DataPipeline
from tytorch.strategies.global_transform_strategies import ImageTensorSplitStrategy
from tytorch.strategies.item_transform_strategies import ImageTensorAugmentationStrategy
from tytorch.strategies.loader_strategies import ImageTensorLoaderStrategy

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

# Create and return the datasets
train_dataset, val_dataset, test_dataset = data_pipeline.create_datasets(
    item_transform_strategies
)

# Use the datasets in DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Access batches from the train loader
for data, labels in train_loader:
    print(data.shape, labels.shape)
