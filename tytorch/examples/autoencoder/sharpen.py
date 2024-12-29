from pathlib import Path

import matplotlib.pyplot as plt
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor

from tytorch.examples.autoencoder.utils import VAESettings, VAEstreamer

logdir = Path("models/embeddings/")
writer = SummaryWriter(log_dir=logdir)

logger.add("/tmp/autoencoder.log")
logger.add("logs/vae.log")


def main():
    logger.info("starting vae_embeddings.py")

    presets = VAESettings()

    test_data = datasets.MNIST(
        root=presets.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )
    teststreamer = VAEstreamer(test_data, batchsize=1).stream(channel_first=True)

    modelpath = presets.modeldir / presets.modelname
    logger.info(f"loading pretrained model {modelpath}")
    model = torch.load(modelpath)
    X, Y = next(teststreamer)

    embs = model.encoder(X.to("mps"))

    Y_hat = model.decoder(embs)

    Y_hat_cpu = Y_hat.cpu().detach().numpy()

    # Set up a plot to show a few images
    num_images_to_show = 5  # Number of images to display
    fig, axes = plt.subplots(3, 1, figsize=(15, 5))

    # Display blurred image
    axes[0].imshow(X.squeeze())
    axes[0].axis("off")
    axes[0].set_title("blurred")

    # Display pred image
    axes[1].imshow(Y_hat_cpu.squeeze())
    axes[1].axis("off")
    axes[1].set_title("sharpened")

    # Display pred image
    axes[2].imshow(Y.squeeze())
    axes[2].axis("off")
    axes[2].set_title("original")

    plt.show()


if __name__ == "__main__":
    main()
