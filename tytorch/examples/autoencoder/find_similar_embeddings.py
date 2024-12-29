from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from scipy.spatial import KDTree
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

from tytorch.examples.autoencoder.utils import VAESettings

logger.add("logs/vae.log")


def plot_grid(
    img: np.ndarray,
    filepath: Path,
    k: int = 3,
    figsize: tuple = (10, 10),
    title: str = "",
) -> None:
    fig, axs = plt.subplots(k, k, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    axs = axs.ravel()
    for i in tqdm(range(k * k)):
        axs[i].imshow(img[i], cmap="gray")
        axs[i].axis("off")
    fig.savefig(filepath)
    logger.success(f"saved grid to {filepath}")


def main():
    presets = VAESettings()
    embedfile = "models/embeds.pt"

    img, embeds = torch.load(embedfile)
    logger.info(f"Loaded {embedfile} with shape {embeds.shape}")
    kdtree = KDTree(embeds)

    test_data = datasets.MNIST(
        root=presets.data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    modelpath = presets.modeldir / presets.modelname
    logger.info(f"loading pretrained model {modelpath}")
    model = torch.load(modelpath)

    x, y = test_data[1]

    other = model.encoder(x.to("mps"))

    dd, ii = kdtree.query(other.detach().cpu().numpy(), k=9)

    closest = img[ii[0]]
    logger.info(f"closest items for label {y}")
    imgpath = presets.imgpath / Path(f"closest-label-{y}.png")
    plot_grid(closest, imgpath, title=f"Label {y}")


if __name__ == "__main__":
    main()
