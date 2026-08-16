"""NAFNet-style architecture for image restoration with 2x super-resolution.

Based on: "Simple Baselines for Image Restoration" (arXiv:2204.04676)
Adapted for:
- Grayscale input (1 channel)
- 2x upsampling (128x128 -> 256x256)
- Optional noise conditioning (none/scalar/spatial)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SimpleGate(nn.Module):
    """Simple gating mechanism - splits channels and multiplies."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Simplified channel attention using global average pooling."""
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class NAFBlock(nn.Module):
    """NAFNet basic block with SimpleGate and simplified attention."""

    def __init__(self, channels: int, dropout_rate: float = 0.0):
        super().__init__()
        dw_channels = channels * 2

        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels, bias=True)
        self.gate = SimpleGate()
        self.sca = SimplifiedChannelAttention(channels)
        self.conv3 = nn.Conv2d(channels, channels, 1, bias=True)

        self.norm2 = nn.LayerNorm(channels)
        self.conv4 = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.conv5 = nn.Conv2d(channels, channels, 1, bias=True)

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Learnable scaling factors
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial attention path
        b, c, h, w = x.shape
        residual = x

        # LayerNorm expects (B, C, H, W) -> permute for norm
        out = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = self.norm1(out).permute(0, 3, 1, 2)  # (B, C, H, W)

        out = self.conv1(out)       # channels -> dw_channels
        out = self.conv2(out)       # depthwise conv on dw_channels
        out = self.gate(out)        # dw_channels -> channels (SimpleGate halves)
        out = self.sca(out)
        out = self.conv3(out)
        out = self.dropout(out)

        x = residual + out * self.beta

        # Channel attention path (FFN)
        residual = x
        out = x.permute(0, 2, 3, 1)
        out = self.norm2(out).permute(0, 3, 1, 2)
        out = self.conv4(out)
        out = self.gate(out)
        out = self.conv5(out)
        out = self.dropout(out)

        return residual + out * self.gamma


class NAFNet(nn.Module):
    """NAFNet for image restoration with 2x super-resolution.

    Architecture:
        Encoder -> Middle -> Decoder -> 2x Pixel Shuffle upsampling
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        num_blocks: List[int] = [2, 4, 8, 8],
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.width = width

        # Initial feature extraction
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1, bias=True)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in num_blocks[:-1]:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, dropout_rate) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2, bias=True))
            chan *= 2

        # Middle
        self.middle = nn.Sequential(
            *[NAFBlock(chan, dropout_rate) for _ in range(num_blocks[-1])]
        )

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i, num in enumerate(reversed(num_blocks[:-1])):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=True),
                    nn.PixelShuffle(2),
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan, dropout_rate) for _ in range(num)])
            )

        # 2x upsampling for super-resolution
        self.upsample = nn.Sequential(
            nn.Conv2d(width, width * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.Conv2d(width, out_channels, 3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, 1, 128, 128)

        Returns:
            Output (B, 1, 256, 256) clamped to [0, 1]
        """
        feat = self.intro(x)

        # Encoder
        enc_features = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            enc_features.append(feat)
            feat = down(feat)

        # Middle
        feat = self.middle(feat)

        # Decoder with skip connections
        for decoder, up, enc_feat in zip(
            self.decoders, self.ups, reversed(enc_features)
        ):
            feat = up(feat)
            feat = feat + enc_feat
            feat = decoder(feat)

        # 2x super-resolution upsampling
        out = self.upsample(feat)

        return torch.clamp(out, 0.0, 1.0)


class NoiseEstimator(nn.Module):
    """Estimates spatial noise map from input image."""

    def __init__(self, in_channels: int = 1, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, in_channels, 3, padding=1),
            nn.Softplus(),  # Ensure positive noise estimate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoiseAwareNAFNet(nn.Module):
    """NAFNet with noise conditioning for improved degradation handling.

    Supports three noise conditioning modes:
    - none: Standard NAFNet
    - scalar: Global noise level estimate concatenated as constant map
    - spatial: Pixel-wise noise map concatenated to input
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        num_blocks: List[int] = [2, 4, 8, 8],
        dropout_rate: float = 0.0,
        noise_mode: str = "spatial",
    ):
        super().__init__()
        self.noise_mode = noise_mode

        # Noise estimator
        if noise_mode != "none":
            self.noise_estimator = NoiseEstimator(in_channels, width=32)
            nafnet_in = in_channels + 1  # concatenate noise map
        else:
            self.noise_estimator = None
            nafnet_in = in_channels

        # Main restoration network
        self.nafnet = NAFNet(
            in_channels=nafnet_in,
            out_channels=out_channels,
            width=width,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, 1, 128, 128) - raw NoisyLR, NOT clipped.

        Returns:
            Output (B, 1, 256, 256) clamped to [0, 1].
        """
        if self.noise_mode == "none":
            return self.nafnet(x)

        # Estimate noise
        noise_map = self.noise_estimator(x)

        if self.noise_mode == "scalar":
            # Use mean noise level as constant spatial map
            noise_level = noise_map.mean(dim=[2, 3], keepdim=True)
            noise_cond = noise_level.expand_as(x)
        else:  # spatial
            noise_cond = noise_map

        # Concatenate noise map with input
        conditioned = torch.cat([x, noise_cond], dim=1)
        return self.nafnet(conditioned)

    def get_noise_map(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get estimated noise map for visualization."""
        if self.noise_estimator is not None:
            return self.noise_estimator(x)
        return None
