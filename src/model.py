"""
Model Architecture: Enhanced ESRGAN with Geospatial Attention Module
=====================================================================

This module implements:
1. ESRGAN Generator with RRDB (Residual-in-Residual Dense Blocks)
2. Geospatial Attention Module (GAM) for land-cover aware processing
3. Multi-Scale Discriminator for hallucination detection at multiple frequencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


# =============================================================================
# Building Blocks
# =============================================================================

class DenseBlock(nn.Module):
    """Dense Block with growth rate for feature extraction."""

    def __init__(self, in_channels: int, growth_rate: int = 32, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        self.growth_rate = growth_rate
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block - core building block of ESRGAN."""

    def __init__(self, channels: int, growth_rate: int = 32, num_dense_blocks: int = 3,
                 residual_scaling: float = 0.2):
        super().__init__()
        self.residual_scaling = residual_scaling

        # Calculate output channels from dense blocks
        dense_out_channels = channels + 4 * growth_rate  # 4 layers per dense block

        self.dense_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        for _ in range(num_dense_blocks):
            self.dense_blocks.append(DenseBlock(channels, growth_rate))
            self.conv_blocks.append(
                nn.Conv2d(dense_out_channels, channels, 1, 1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for dense, conv in zip(self.dense_blocks, self.conv_blocks):
            residual = out
            out = dense(out)
            out = conv(out)
            out = residual + out * self.residual_scaling
        return x + out * self.residual_scaling


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GeospatialAttentionModule(nn.Module):
    """
    Geospatial Attention Module (GAM) - UNIQUE FEATURE

    Injects land-cover priors (NDVI, NDBI, NDWI) into the generator to weight 
    reconstruction quality based on land-cover type. Forests don't need sharp 
    edges; roads do.

    Input: 16 channels (13 Sentinel-2 bands + 3 computed indices)
    Output: Attention-weighted features
    """

    def __init__(self, in_channels: int = 16, out_channels: int = 64):
        super().__init__()

        # Process geospatial indices separately
        self.index_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 3 indices: NDVI, NDBI, NDWI
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Process spectral bands
        self.band_encoder = nn.Sequential(
            nn.Conv2d(13, 32, 3, 1, 1),  # 13 Sentinel-2 bands
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Attention generation from indices
        self.attention_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Sigmoid()
        )

        # Combine features
        self.fusion = nn.Sequential(
            nn.Conv2d(128, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # SE block for channel attention
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 16, H, W)
               First 13 channels: Sentinel-2 bands
               Last 3 channels: NDVI, NDBI, NDWI indices

        Returns:
            Attention-weighted features of shape (B, out_channels, H, W)
        """
        # Split input into bands and indices
        bands = x[:, :13, :, :]  # Sentinel-2 bands
        indices = x[:, 13:, :, :]  # Geospatial indices

        # Encode separately
        band_features = self.band_encoder(bands)
        index_features = self.index_encoder(indices)

        # Generate spatial attention from indices
        attention = self.attention_conv(index_features)

        # Apply attention to band features
        attended_bands = band_features * attention

        # Fuse all features
        combined = torch.cat([attended_bands, index_features], dim=1)
        out = self.fusion(combined)

        # Apply channel attention
        out = self.se(out)

        return out


class PixelShuffleUpsampler(nn.Module):
    """Efficient upsampling using PixelShuffle (sub-pixel convolution)."""

    def __init__(self, in_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels *
                              (scale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.pixel_shuffle(self.conv(x)))


# =============================================================================
# Generator Network
# =============================================================================

class ESRGANGenerator(nn.Module):
    """
    Enhanced ESRGAN Generator with Geospatial Attention Module

    Architecture:
    - Input: 16-channel (13 Sentinel-2 bands + 3 indices) at LR resolution
    - GAM: Encode geospatial priors
    - RRDB trunk: 23 RRDB blocks for feature extraction
    - Progressive upsampling: 2x → 2x → 2x (8x total)
    - Output: 3-channel RGB at HR resolution (1.25m/pixel)
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        num_features: int = 64,
        num_rrdb_blocks: int = 23,
        growth_rate: int = 32,
        scale_factor: int = 8
    ):
        super().__init__()

        self.scale_factor = scale_factor

        # Geospatial Attention Module
        self.gam = GeospatialAttentionModule(in_channels, num_features)

        # First convolution
        self.conv_first = nn.Conv2d(num_features, num_features, 3, 1, 1)

        # RRDB trunk
        self.rrdb_trunk = nn.Sequential(
            *[RRDB(num_features, growth_rate) for _ in range(num_rrdb_blocks)]
        )

        # Trunk convolution
        self.trunk_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)

        # Progressive upsampling (2x → 2x → 2x = 8x)
        self.upsampler = nn.Sequential(
            PixelShuffleUpsampler(num_features, 2),  # 2x
            PixelShuffleUpsampler(num_features, 2),  # 4x
            PixelShuffleUpsampler(num_features, 2),  # 8x
        )

        # High-resolution convolutions
        self.hr_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LR input (B, 16, H, W) - 13 bands + 3 indices

        Returns:
            SR output (B, 3, H*8, W*8) - RGB at 8x resolution
        """
        # Geospatial attention encoding
        feat = self.gam(x)

        # First convolution
        feat = self.conv_first(feat)

        # RRDB trunk with residual connection
        trunk = self.trunk_conv(self.rrdb_trunk(feat))
        feat = feat + trunk

        # Progressive upsampling
        feat = self.upsampler(feat)

        # HR output
        out = self.hr_conv(feat)

        return out


class ESRGANGeneratorRGB(nn.Module):
    """
    Simplified ESRGAN Generator for RGB-only input (3 channels).
    Useful for training on standard SR datasets or when indices aren't available.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_rrdb_blocks: int = 23,
        growth_rate: int = 32,
        scale_factor: int = 8
    ):
        super().__init__()

        self.scale_factor = scale_factor

        # First convolution
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        # RRDB trunk
        self.rrdb_trunk = nn.Sequential(
            *[RRDB(num_features, growth_rate) for _ in range(num_rrdb_blocks)]
        )

        # Trunk convolution
        self.trunk_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)

        # Progressive upsampling (2x → 2x → 2x = 8x)
        self.upsampler = nn.Sequential(
            PixelShuffleUpsampler(num_features, 2),  # 2x
            PixelShuffleUpsampler(num_features, 2),  # 4x
            PixelShuffleUpsampler(num_features, 2),  # 8x
        )

        # High-resolution convolutions
        self.hr_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First convolution
        feat = self.conv_first(x)

        # RRDB trunk with residual connection
        trunk = self.trunk_conv(self.rrdb_trunk(feat))
        feat = feat + trunk

        # Progressive upsampling
        feat = self.upsampler(feat)

        # HR output
        out = self.hr_conv(feat)

        return out


# =============================================================================
# Multi-Scale Discriminator
# =============================================================================

class SpectralNorm(nn.Module):
    """Spectral Normalization wrapper for training stability."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = nn.utils.spectral_norm(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with spectral normalization.

    Classifies 70x70 patches as real/fake instead of entire image.
    Better for high-resolution image generation.
    """

    def __init__(self, in_channels: int = 3, num_features: int = 64):
        super().__init__()

        def discriminator_block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, 4, stride, 1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, num_features, normalize=False),
            *discriminator_block(num_features, num_features * 2),
            *discriminator_block(num_features * 2, num_features * 4),
            *discriminator_block(num_features * 4, num_features * 8, stride=1),
            nn.utils.spectral_norm(nn.Conv2d(num_features * 8, 1, 4, 1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator - INNOVATION

    Trains 3 discriminators at different scales to catch hallucinations
    at multiple spatial frequencies:

    - D₁ (1x): Full resolution - validates fine details (road edges, building corners)
    - D₂ (0.5x): Half resolution - validates mid-level structures (parking lots, rooftops)  
    - D₃ (0.25x): Quarter resolution - validates macro patterns (urban vs vegetation)
    """

    def __init__(self, in_channels: int = 3, num_features: int = 64, num_discriminators: int = 3):
        super().__init__()

        self.num_discriminators = num_discriminators
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, num_features)
            for _ in range(num_discriminators)
        ])

        # Downsampling for multi-scale
        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input image (B, C, H, W)

        Returns:
            List of discriminator outputs at different scales
        """
        outputs = []
        input_downsampled = x

        for i, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(input_downsampled))
            if i < self.num_discriminators - 1:
                input_downsampled = self.downsample(input_downsampled)

        return outputs


# =============================================================================
# VGG Feature Extractor for Perceptual Loss
# =============================================================================

class VGGFeatureExtractor(nn.Module):
    """
    VGG19 feature extractor for perceptual loss.
    Extracts features from conv5_4 layer to match textures.
    """

    def __init__(self, feature_layer: int = 35, use_bn: bool = False, use_input_norm: bool = True):
        super().__init__()

        if use_bn:
            from torchvision.models import vgg19_bn, VGG19_BN_Weights
            model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
        else:
            from torchvision.models import vgg19, VGG19_Weights
            model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        self.use_input_norm = use_input_norm

        if self.use_input_norm:
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        # Extract features up to specified layer
        self.features = nn.Sequential(
            *list(model.features.children())[:feature_layer + 1])

        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.features(x)


# =============================================================================
# Model Factory Functions
# =============================================================================

def create_generator(
    use_gam: bool = True,
    in_channels: int = 16,
    num_rrdb_blocks: int = 23,
    scale_factor: int = 8
) -> nn.Module:
    """
    Factory function to create generator.

    Args:
        use_gam: Whether to use Geospatial Attention Module
        in_channels: Number of input channels (16 with indices, 3 for RGB)
        num_rrdb_blocks: Number of RRDB blocks (23 is standard)
        scale_factor: Upscaling factor (8 for 10m → 1.25m)

    Returns:
        Generator model
    """
    if use_gam and in_channels == 16:
        return ESRGANGenerator(
            in_channels=in_channels,
            num_rrdb_blocks=num_rrdb_blocks,
            scale_factor=scale_factor
        )
    else:
        return ESRGANGeneratorRGB(
            in_channels=in_channels if in_channels != 16 else 3,
            num_rrdb_blocks=num_rrdb_blocks,
            scale_factor=scale_factor
        )


def create_discriminator(num_discriminators: int = 3) -> nn.Module:
    """
    Factory function to create multi-scale discriminator.

    Args:
        num_discriminators: Number of discriminator scales (default: 3)

    Returns:
        Multi-scale discriminator model
    """
    return MultiScaleDiscriminator(num_discriminators=num_discriminators)


def create_vgg_extractor() -> nn.Module:
    """Create VGG feature extractor for perceptual loss."""
    return VGGFeatureExtractor()


# =============================================================================
# Model Summary
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary():
    """Print model architecture summary."""
    print("=" * 60)
    print("KLYMO ASCENT ML - Model Architecture Summary")
    print("=" * 60)

    # Generator with GAM
    gen = create_generator(use_gam=True)
    print(f"\nGenerator (ESRGAN + GAM):")
    print(f"  - Input: 16 channels (13 bands + 3 indices)")
    print(f"  - Output: 3 channels (RGB)")
    print(f"  - Scale Factor: 8x")
    print(f"  - Parameters: {count_parameters(gen):,}")

    # Generator without GAM
    gen_rgb = create_generator(use_gam=False, in_channels=3)
    print(f"\nGenerator (ESRGAN RGB only):")
    print(f"  - Input: 3 channels")
    print(f"  - Output: 3 channels")
    print(f"  - Parameters: {count_parameters(gen_rgb):,}")

    # Discriminator
    disc = create_discriminator()
    print(f"\nMulti-Scale Discriminator:")
    print(f"  - Scales: 3 (1x, 0.5x, 0.25x)")
    print(f"  - Parameters: {count_parameters(disc):,}")

    print("=" * 60)


if __name__ == "__main__":
    model_summary()

    # Test forward pass
    print("\nTesting forward pass...")

    # Generator with GAM
    gen = create_generator(use_gam=True)
    x = torch.randn(1, 16, 32, 32)
    y = gen(x)
    print(f"Generator: {x.shape} → {y.shape}")

    # Discriminator
    disc = create_discriminator()
    outputs = disc(y)
    print(f"Discriminator outputs: {[o.shape for o in outputs]}")

    print("\n✓ All tests passed!")
