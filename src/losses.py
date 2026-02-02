"""
Loss Functions for Geospatial Super-Resolution
================================================

This module implements the composite loss function:

Total Loss = λ₁·L_pixel + λ₂·L_perceptual + λ₃·L_adversarial + λ₄·L_spatial_consistency

Key Innovation: Spatial Consistency Loss prevents hallucination by penalizing
features not present in the bicubic upsampled low-resolution input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


# =============================================================================
# Pixel-wise Losses
# =============================================================================

class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss for pixel-wise accuracy."""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (differentiable L1 variant).
    Better gradients near zero compared to standard L1.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.epsilon))


# =============================================================================
# Perceptual Loss (VGG-based)
# =============================================================================

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG19 features.

    Compares high-level feature representations instead of pixels,
    resulting in more perceptually pleasing outputs.
    """

    def __init__(self, feature_layer: int = 35, use_input_norm: bool = True):
        super().__init__()

        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        self.use_input_norm = use_input_norm

        if self.use_input_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        # Extract features up to specified layer (conv5_4 = layer 35)
        self.features = nn.Sequential(
            *list(vgg.features.children())[:feature_layer + 1])

        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_input_norm:
            pred = (pred - self.mean.to(pred.device)) / \
                self.std.to(pred.device)
            target = (target - self.mean.to(target.device)) / \
                self.std.to(target.device)

        pred_features = self.features(pred)
        target_features = self.features(target)

        return self.criterion(pred_features, target_features)


# =============================================================================
# Adversarial Losses
# =============================================================================

class GANLoss(nn.Module):
    """
    GAN Loss for generator and discriminator.

    Supports multiple loss types:
    - 'vanilla': Binary cross-entropy
    - 'lsgan': Least squares GAN (more stable training)
    - 'wgan': Wasserstein GAN
    - 'wgan-gp': Wasserstein GAN with gradient penalty
    """

    def __init__(self, gan_type: str = 'vanilla', real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.gan_type = gan_type
        self.real_label = real_label
        self.fake_label = fake_label

        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type in ['wgan', 'wgan-gp']:
            self.loss = None  # Uses Wasserstein distance directly
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")

    def _get_target_tensor(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        target_value = self.real_label if is_real else self.fake_label
        return torch.full_like(prediction, target_value)

    def forward(
        self,
        prediction: torch.Tensor,
        is_real: bool,
        for_discriminator: bool = True
    ) -> torch.Tensor:
        """
        Calculate GAN loss.

        Args:
            prediction: Discriminator output
            is_real: Whether the input is real or fake
            for_discriminator: Whether computing loss for D (True) or G (False)

        Returns:
            GAN loss value
        """
        if self.gan_type in ['wgan', 'wgan-gp']:
            if for_discriminator:
                return -prediction.mean() if is_real else prediction.mean()
            else:
                return -prediction.mean()  # Generator wants D(G(z)) to be high
        else:
            target = self._get_target_tensor(prediction, is_real)
            return self.loss(prediction, target)


class MultiScaleGANLoss(nn.Module):
    """
    GAN Loss for Multi-Scale Discriminator.
    Aggregates losses from all discriminator scales.
    """

    def __init__(self, gan_type: str = 'vanilla', num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.gan_losses = nn.ModuleList([
            GANLoss(gan_type) for _ in range(num_scales)
        ])

        # Scale weights (higher resolution = more important)
        self.scale_weights = [1.0, 0.5, 0.25]

    def forward(
        self,
        predictions: List[torch.Tensor],
        is_real: bool,
        for_discriminator: bool = True
    ) -> torch.Tensor:
        """
        Calculate multi-scale GAN loss.

        Args:
            predictions: List of discriminator outputs at different scales
            is_real: Whether the input is real or fake
            for_discriminator: Whether computing loss for D or G

        Returns:
            Weighted sum of GAN losses across scales
        """
        total_loss = 0.0
        for i, (pred, gan_loss) in enumerate(zip(predictions, self.gan_losses)):
            weight = self.scale_weights[i] if i < len(
                self.scale_weights) else 1.0
            total_loss += weight * gan_loss(pred, is_real, for_discriminator)

        return total_loss / sum(self.scale_weights[:len(predictions)])


# =============================================================================
# Spatial Consistency Loss - HALLUCINATION PREVENTION (KEY INNOVATION)
# =============================================================================

class SobelFilter(nn.Module):
    """Sobel filter for edge detection."""

    def __init__(self):
        super().__init__()

        # Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Sobel filter to detect edges.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Gradient magnitude (B, C, H, W)
        """
        # Process each channel separately
        batch_size, channels, height, width = x.shape

        # Expand kernels for all channels
        sobel_x = self.sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(channels, 1, 1, 1)

        # Apply Sobel filters
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)

        # Gradient magnitude
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        return gradient


class SpatialConsistencyLoss(nn.Module):
    """
    Spatial Consistency Loss - HALLUCINATION PREVENTION

    Mathematical formulation:
    L_spatial = ||∇(SR) - ∇(Bicubic_8x(LR))||₂ + ||FFT(SR) - FFT(Bicubic_8x(LR))||₂

    This loss ensures:
    1. Edges in SR output align with edges in bicubic baseline (no invented edges)
    2. Frequency spectrum matches baseline (no texture hallucination)

    If a building edge isn't visible in the bicubic upsampled version,
    the model shouldn't invent it.
    """

    def __init__(
        self,
        gradient_weight: float = 1.0,
        fft_weight: float = 0.5,
        scale_factor: int = 8
    ):
        super().__init__()

        self.gradient_weight = gradient_weight
        self.fft_weight = fft_weight
        self.scale_factor = scale_factor
        self.sobel = SobelFilter()

    def _compute_fft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute FFT-based frequency loss."""
        # Apply 2D FFT
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        # Get magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # L1 loss on magnitude spectrum
        return F.l1_loss(pred_mag, target_mag)

    def forward(
        self,
        sr_output: torch.Tensor,
        lr_input: torch.Tensor,
        use_fft: bool = True
    ) -> torch.Tensor:
        """
        Calculate spatial consistency loss.

        Args:
            sr_output: Super-resolved output (B, 3, H*8, W*8)
            lr_input: Low-resolution input (B, 3, H, W) - RGB channels only
            use_fft: Whether to include FFT loss component

        Returns:
            Spatial consistency loss value
        """
        # Generate bicubic baseline (what we know exists in LR)
        bicubic_baseline = F.interpolate(
            lr_input,
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )

        # Gradient loss - ensure edge alignment
        sr_gradient = self.sobel(sr_output)
        baseline_gradient = self.sobel(bicubic_baseline)
        gradient_loss = F.l1_loss(sr_gradient, baseline_gradient)

        total_loss = self.gradient_weight * gradient_loss

        # FFT loss - ensure frequency spectrum match
        if use_fft:
            fft_loss = self._compute_fft_loss(sr_output, bicubic_baseline)
            total_loss += self.fft_weight * fft_loss

        return total_loss


class EdgeAlignmentLoss(nn.Module):
    """
    Alternative edge-based loss using Canny-like detection.
    Penalizes edges in SR that don't exist in bicubic baseline.
    """

    def __init__(self, scale_factor: int = 8):
        super().__init__()
        self.scale_factor = scale_factor
        self.sobel = SobelFilter()

    def forward(
        self,
        sr_output: torch.Tensor,
        lr_input: torch.Tensor
    ) -> torch.Tensor:
        # Bicubic baseline
        bicubic = F.interpolate(
            lr_input,
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )

        # Edge maps
        sr_edges = self.sobel(sr_output)
        bicubic_edges = self.sobel(bicubic)

        # Threshold edges
        sr_edge_binary = (sr_edges > 0.1).float()
        # Lower threshold for baseline
        bicubic_edge_binary = (bicubic_edges > 0.05).float()

        # Penalize edges in SR that don't exist in bicubic
        # (inverted mask multiplication)
        hallucinated_edges = sr_edge_binary * (1 - bicubic_edge_binary)

        return hallucinated_edges.mean()


# =============================================================================
# Composite Loss Function
# =============================================================================

class KlymoLoss(nn.Module):
    """
    Complete loss function for KLYMO Geospatial Super-Resolution.

    Total Loss = λ₁·L_pixel + λ₂·L_perceptual + λ₃·L_adversarial + λ₄·L_spatial_consistency

    Default weights from PRD:
    - λ₁ (pixel): 1.0
    - λ₂ (perceptual): 0.1
    - λ₃ (adversarial): 0.005
    - λ₄ (spatial_consistency): 0.5 (HIGH - prevents hallucination)
    """

    def __init__(
        self,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        adversarial_weight: float = 0.005,
        spatial_weight: float = 0.5,
        use_charbonnier: bool = True,
        gan_type: str = 'vanilla',
        num_discriminator_scales: int = 3
    ):
        super().__init__()

        # Loss weights
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.spatial_weight = spatial_weight

        # Individual losses
        self.pixel_loss = CharbonnierLoss() if use_charbonnier else L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = MultiScaleGANLoss(
            gan_type, num_discriminator_scales)
        self.spatial_loss = SpatialConsistencyLoss()

    def forward(
        self,
        sr_output: torch.Tensor,
        hr_target: torch.Tensor,
        lr_input: torch.Tensor,
        disc_fake_outputs: Optional[List[torch.Tensor]] = None,
        phase: str = 'psnr'
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate composite loss.

        Args:
            sr_output: Super-resolved output (B, 3, H*8, W*8)
            hr_target: High-resolution ground truth (B, 3, H*8, W*8)
            lr_input: Low-resolution input RGB (B, 3, H, W)
            disc_fake_outputs: Discriminator outputs for fake (SR) images
            phase: Training phase ('psnr' or 'gan')

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # Pixel loss (always used)
        l_pixel = self.pixel_loss(sr_output, hr_target)
        loss_dict['l_pixel'] = l_pixel.item()
        total_loss += self.pixel_weight * l_pixel

        # Spatial consistency loss (always used - prevents hallucination)
        l_spatial = self.spatial_loss(sr_output, lr_input)
        loss_dict['l_spatial'] = l_spatial.item()
        total_loss += self.spatial_weight * l_spatial

        # Phase 2 (GAN) specific losses
        if phase == 'gan':
            # Perceptual loss
            l_perceptual = self.perceptual_loss(sr_output, hr_target)
            loss_dict['l_perceptual'] = l_perceptual.item()
            total_loss += self.perceptual_weight * l_perceptual

            # Adversarial loss (generator)
            if disc_fake_outputs is not None:
                l_adversarial = self.adversarial_loss(
                    disc_fake_outputs,
                    is_real=True,  # Generator wants D to think fake is real
                    for_discriminator=False
                )
                loss_dict['l_adversarial'] = l_adversarial.item()
                total_loss += self.adversarial_weight * l_adversarial

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def discriminator_loss(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_fake_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate discriminator loss.

        Args:
            disc_real_outputs: Discriminator outputs for real (HR) images
            disc_fake_outputs: Discriminator outputs for fake (SR) images

        Returns:
            d_loss: Discriminator loss
            loss_dict: Dictionary with loss components
        """
        # Real loss
        loss_real = self.adversarial_loss(
            disc_real_outputs,
            is_real=True,
            for_discriminator=True
        )

        # Fake loss
        loss_fake = self.adversarial_loss(
            disc_fake_outputs,
            is_real=False,
            for_discriminator=True
        )

        d_loss = (loss_real + loss_fake) / 2

        loss_dict = {
            'd_real': loss_real.item(),
            'd_fake': loss_fake.item(),
            'd_total': d_loss.item()
        }

        return d_loss, loss_dict


# =============================================================================
# Loss Configuration for Different Training Phases
# =============================================================================

def get_phase1_loss() -> KlymoLoss:
    """
    Phase 1: PSNR Pre-training
    Only pixel + spatial consistency loss (no GAN)
    """
    return KlymoLoss(
        pixel_weight=1.0,
        perceptual_weight=0.0,
        adversarial_weight=0.0,
        spatial_weight=0.5
    )


def get_phase2_loss() -> KlymoLoss:
    """
    Phase 2: GAN Fine-tuning
    All loss components active
    """
    return KlymoLoss(
        pixel_weight=1.0,
        perceptual_weight=0.1,
        adversarial_weight=0.005,
        spatial_weight=0.5
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    # Create dummy data
    batch_size = 2
    lr_size = 32
    hr_size = lr_size * 8  # 256

    lr_input = torch.randn(batch_size, 3, lr_size, lr_size)
    sr_output = torch.randn(batch_size, 3, hr_size, hr_size)
    hr_target = torch.randn(batch_size, 3, hr_size, hr_size)

    # Test individual losses
    print("\nIndividual Losses:")

    pixel = CharbonnierLoss()
    print(f"  Charbonnier: {pixel(sr_output, hr_target).item():.4f}")

    spatial = SpatialConsistencyLoss()
    print(f"  Spatial Consistency: {spatial(sr_output, lr_input).item():.4f}")

    # Test composite loss (Phase 1)
    print("\nPhase 1 Loss (PSNR):")
    loss_fn = get_phase1_loss()
    total, loss_dict = loss_fn(sr_output, hr_target, lr_input, phase='psnr')
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    # Test composite loss (Phase 2)
    print("\nPhase 2 Loss (GAN):")
    loss_fn = get_phase2_loss()

    # Simulate discriminator outputs
    disc_outputs = [
        torch.randn(batch_size, 1, 16, 16),
        torch.randn(batch_size, 1, 8, 8),
        torch.randn(batch_size, 1, 4, 4)
    ]

    total, loss_dict = loss_fn(
        sr_output, hr_target, lr_input, disc_outputs, phase='gan')
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ All loss tests passed!")
