from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F



@dataclass
class VAESettings:
    data_dir: Path = Path("data")
    h1: int = 250
    h2: int = 100
    insize: int = 784
    latent: int = 32
    batchsize: int = 32
    epochs: int = 100
    modelname: Path = Path("vaemodel.pt")
    modeldir: Path = Path("models")
    imgpath: Path = Path("img")
    samplesize: int = 512


class VAEstreamer(BaseDatastreamer):
    def stream(self, channel_first: bool = None) -> Iterator:
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            # we throw away the Y
            X_, _ = zip(*batch)  # noqa N806
            X = torch.stack(X_)  # noqa N806
            # change the channel to channel-last
            X = torch.moveaxis(X, 1, 3)  # noqa N806
            # and yield X, X

            # Apply blur to all images in X
            X_blurred = F.avg_pool2d(X, kernel_size=(21, 21), stride=1, padding=10)
            if channel_first:
                X_blurred = X_blurred.permute(0, 3, 1, 2)
                X = X.permute(0, 3, 1, 2)

            # Yield the blurred images as input and the original images as target
            yield X_blurred, X
