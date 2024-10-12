from typing import Any, Dict, Optional, Tuple

from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        """Convolutional block with optional dropout and maxpool."""
        super(ConvBlock, self).__init__()
        self.pool = pool
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]

        if self.pool:
            layers.append(nn.MaxPool2d(kernel_size=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvFlexible(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_conv_layers: int,
        initial_filters: int,
        growth_factor: int = 1,
        pool: bool = True,
        residual: bool = False,
    ) -> None:
        super(ConvFlexible, self).__init__()
        self.input_size = input_size
        self.num_conv_layers = num_conv_layers
        self.initial_filters = initial_filters
        self.growth_factor = growth_factor
        self.pool = pool
        self.residual = residual
        self.convolutions = nn.ModuleList()
        self.spatial_reduction = 1

        for i in range(num_conv_layers):
            if i == 0:
                in_ch = input_size
            else:
                in_ch = initial_filters * (growth_factor ** (i - 1))

            out_ch = (
                initial_filters * (growth_factor**i)
                if growth_factor != 1
                else initial_filters
            )

            self.convolutions.append(
                ConvBlock(in_channels=in_ch, out_channels=out_ch, pool=self.pool)
            )
            if self.pool:
                self.spatial_reduction *= 2

        if self.residual and (input_size != out_ch or self.spatial_reduction > 1):
            self.downsample = nn.Conv2d(
                input_size,
                out_ch,
                kernel_size=1,
                stride=self.spatial_reduction,
                padding=1,
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        for conv in self.convolutions:
            x = conv(x)

        if self.residual:
            if self.downsample is not None:
                identity = self.downsample(identity)

            if identity.shape[-2:] != x.shape[-2:]:
                identity = self._align_dimensions(identity, x.shape[-2:])

            x += identity

        return x

    def _align_dimensions(
        self, identity: Tensor, target_shape: Tuple[int, int]
    ) -> Tensor:
        _, _, h, w = identity.shape
        target_h, target_w = target_shape

        if h > target_h or w > target_w:
            identity = identity[:, :, :target_h, :target_w]

        if h < target_h or w < target_w:
            pad_h = (target_h - h) // 2
            pad_w = (target_w - w) // 2
            identity = nn.functional.pad(identity, (pad_w, pad_w, pad_h, pad_h))

        return identity


class LinearBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, dropout: float = 0.0
    ) -> None:
        """Linear block with ReLU activation and optional dropout."""
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(in_features, out_features), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class CNN(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.input_size = config["input_size"]

        self.convolutions = nn.ModuleList()
        self.last_out_channels: Optional[int] = None
        for i, blockconfig in enumerate(config["conv_blocks"]):
            self.convolutions.append(
                ConvFlexible(
                    self.input_size[1] if i == 0 else self.last_out_channels,
                    blockconfig["num_conv_layers"],
                    blockconfig["initial_filters"],
                    blockconfig["growth_factor"],
                    blockconfig["pool"],
                    blockconfig["residual"],
                )
            )

            self.last_out_channels, self.spatial_dim = self.calculate_output_size(
                self.input_size[1] if i == 0 else self.last_out_channels,
                blockconfig["num_conv_layers"],
                blockconfig["initial_filters"],
                blockconfig["growth_factor"],
                blockconfig["pool"],
            )

        self.dropout = nn.Dropout(config["dropout"])

        self.dense_layers = nn.ModuleList()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        for i, blockconfig in enumerate(config["linear_blocks"]):
            in_features = (
                self.last_out_channels
                if i == 0
                else config["linear_blocks"][i - 1]["out_features"]
            )
            self.dense_layers.append(
                LinearBlock(
                    in_features=in_features,
                    out_features=blockconfig["out_features"],
                    dropout=blockconfig.get("dropout", 0.0),
                )
            )

        self.output_layer = nn.Linear(
            config["linear_blocks"][-1]["out_features"], config["output_size"]
        )

    def calculate_output_size(
        self,
        input_size: int,
        num_conv_layers: int,
        initial_filters: int,
        growth_factor: int,
        pool: bool = True,
    ) -> Tuple[int, int]:
        spatial_dim = input_size
        filters = initial_filters

        for i in range(num_conv_layers):
            if growth_factor != 1:
                filters = initial_filters * (growth_factor**i)

            if pool:
                spatial_dim //= 2

        out_ch = filters
        return out_ch, spatial_dim

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convolutions:
            x = conv(x)
        x = self.dropout(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        for dense in self.dense_layers:
            x = dense(x)
        logits = self.output_layer(x)
        return logits
