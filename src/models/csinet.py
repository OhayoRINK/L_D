"""PyTorch CsiNet port following the author's public Keras architecture.

Reference implementation:
https://github.com/sydney222/Python_CsiNet/blob/master/CsiNet_train.py
"""

import torch
from torch import nn


class ConvBNLeakyReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        # Keras BatchNormalization defaults: epsilon=1e-3, momentum=0.99.
        # PyTorch's momentum is the update fraction, hence 0.01 for the same update.
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.activation = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class DecodedResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-3, momentum=0.01)
        self.act1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-3, momentum=0.01)
        self.act2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv3 = nn.Conv2d(16, 2, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(2, eps=1e-3, momentum=0.01)
        self.final_act = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act1(self.bn1(self.conv1(x)))
        residual = self.act2(self.bn2(self.conv2(residual)))
        residual = self.bn3(self.conv3(residual))
        return self.final_act(x + residual)


class CsiNet(nn.Module):
    """CsiNet autoencoder with the standard CR=1/4 codeword dimension 512."""

    def __init__(self, encoded_dim: int = 512, residual_blocks: int = 2) -> None:
        super().__init__()
        self.encoder_conv = ConvBNLeakyReLU(2, 2)
        self.encoder_fc = nn.Linear(2 * 32 * 32, encoded_dim)
        self.decoder_fc = nn.Linear(encoded_dim, 2 * 32 * 32)
        self.residual_blocks = nn.ModuleList(
            DecodedResidualBlock() for _ in range(residual_blocks)
        )
        self.output_conv = nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=True)
        self.output_activation = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_conv(x)
        return self.encoder_fc(x.flatten(start_dim=1))

    def decode(self, codeword: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(codeword).reshape(-1, 2, 32, 32)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_activation(self.output_conv(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
