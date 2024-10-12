import random
from typing import Tuple

import torch
import torchvision.transforms.functional as f

from tytorch.datapipeline import ItemTransformStrategy


class ImageTensorAugmentationStrategy(ItemTransformStrategy):
    def __init__(
        self,
        crop_size: Tuple[int, int] = (224, 224),
        rotation_degrees: Tuple[int, int] = (-20, 20),
    ) -> None:
        """
        Args:
            crop_size: Tuple indicating the size to which the image will be resized and cropped.
            rotation_degrees: Tuple indicating the range of degrees for random rotation.
        """
        self.crop_size = crop_size
        self.rotation_degrees = rotation_degrees

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a series of random augmentations to the input tensor.

        Args:
            x: Input tensor representing the image.

        Returns:
            Transformed tensor.
        """
        # Random Resized Crop
        if random.random() > 0.5:
            x = f.resized_crop(
                x,
                top=0,
                left=0,
                height=self.crop_size[0],
                width=self.crop_size[1],
                size=self.crop_size,
            )

        # Random Horizontal Flip
        if random.random() > 0.5:
            x = f.hflip(x)

        # Random Rotation
        degrees = random.uniform(self.rotation_degrees[0], self.rotation_degrees[1])
        x = f.rotate(x, degrees)

        # Color Jitter
        x = f.adjust_brightness(x, random.uniform(0.8, 1.2))
        x = f.adjust_contrast(x, random.uniform(0.8, 1.2))
        x = f.adjust_saturation(x, random.uniform(0.8, 1.2))
        x = f.adjust_hue(x, random.uniform(-0.1, 0.1))

        return x
