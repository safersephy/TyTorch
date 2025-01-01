from dataclasses import asdict

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision.transforms import ToTensor

from tytorch.examples.autoencoder.utils import VAESettings, VAEstreamer

# from tytorch.examples.models.CustomVAE import AutoEncoder, Encoder, Decoder
from tytorch.examples.models.vae import ConvAutoEncoder, ConvDecoder, ConvEncoder
from tytorch.trainer import Trainer
from tytorch.utils.mlflow import set_mlflow_experiment

logger.add("/tmp/autoencoder.log")
logger.add("logs/vae.log")


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, y, yhat):
        sqe = (y - yhat) ** 2
        if isinstance(sqe, torch.Tensor):
            summed = sqe.sum(dim=(1, 2, 3))
        elif isinstance(sqe, np.ndarray):
            summed = np.sum(sqe, axis=(1, 2, 3))
        else:
            raise TypeError("Input should be either a PyTorch tensor or a NumPy array.")
        return summed.mean()


def sample_range(encoder, stream, k: int = 10):
    minmax_list = []
    for _ in range(10):
        X, _ = next(stream)
        y = encoder(X).detach().numpy()
        minmax_list.append(y.min())
        minmax_list.append(y.max())
    minmax = np.array(minmax_list)
    return minmax.min(), minmax.max()


def build_latent_grid(decoder, minimum: int, maximum: int, k: int = 20):
    x = np.linspace(minimum, maximum, k)
    y = np.linspace(minimum, maximum, k)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]

    img = decoder(torch.tensor(grid, dtype=torch.float32))
    return img.detach().numpy()


def select_n_random(data, labels, n=300):
    """
    Selects n random datapoints and their corresponding labels from a dataset
    """
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


def main():
    params = {
        "batch_size": 32,
        "n_epochs": 100,
        "lr": 1e-4,
    }

    logger.info("starting autoencode.py")

    presets = VAESettings()

    logger.info("loading MNIST datasets")
    training_data = datasets.MNIST(
        root=presets.data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.MNIST(
        root=presets.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    logger.info(
        f"Length trainset: {len(training_data)}, length testset: {len(test_data)}"
    )

    logger.info("creating datastreamers")
    trainstreamer = VAEstreamer(training_data, batchsize=presets.batchsize).stream(
        channel_first=True
    )
    teststreamer = VAEstreamer(test_data, batchsize=32).stream(channel_first=True)

    X1, X2 = next(trainstreamer)

    X_blurred = X1.cpu().detach().numpy()
    X_original = X2.cpu().detach().numpy()

    # Set up a plot to show a few images
    num_images_to_show = 5  # Number of images to display
    fig, axes = plt.subplots(2, num_images_to_show, figsize=(15, 5))

    for i in range(num_images_to_show):
        # Display blurred image
        axes[0, i].imshow(X_blurred[i].squeeze())
        axes[0, i].axis("off")
        axes[0, i].set_title("Blurred")

        # Display original image
        axes[1, i].imshow(X_original[i].squeeze())
        axes[1, i].axis("off")
        axes[1, i].set_title("Original")

    plt.show()

    config = asdict(presets)
    logger.info(f"the shape before: {X1.shape}")

    encoder = ConvEncoder()
    decoder = ConvDecoder()

    latent = encoder(X1)
    logger.info(f"the latent shape : {latent.shape}")

    x = decoder(latent)
    logger.info(f"the shape after: {x.shape}")

    # lossfn = ReconstructionLoss()
    lossfn = MSELoss()
    loss = lossfn(x, X2)
    logger.info(f"Untrained loss: {loss}")

    autoencoder = ConvAutoEncoder(config)

    optimizer = Adam(autoencoder.parameters(), lr=params["lr"])

    trainer = Trainer(
        model=autoencoder,
        loss_fn=lossfn,
        metrics=[],
        optimizer=optimizer,
        device="mps",
        lrscheduler=ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=10),
        train_steps=200,
        valid_steps=200,
    )

    set_mlflow_experiment("train")
    with mlflow.start_run():
        mlflow.log_params(params)
        trainer.fit(params["n_epochs"], trainstreamer, teststreamer)

        mlflow.pytorch.log_model(autoencoder, artifact_path="logged_models/model")
    mlflow.end_run()

    modeldir = presets.modeldir

    if not modeldir.exists():
        modeldir.mkdir(parents=True)

    modelpath = modeldir / presets.modelname

    torch.save(autoencoder, modelpath)

    logger.success("finished autoencode.py")


if __name__ == "__main__":
    main()
