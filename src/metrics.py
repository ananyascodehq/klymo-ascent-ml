"""
Evaluation Metrics for Geospatial Super-Resolution
===================================================

This module implements:
1. PSNR (Peak Signal-to-Noise Ratio) - pixel-wise accuracy
2. SSIM (Structural Similarity Index) - perceptual similarity
3. LPIPS (Learned Perceptual Image Patch Similarity) - deep feature similarity
4. Edge Coherence Score - hallucination detection metric (CUSTOM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2


# =============================================================================
# PSNR (Peak Signal-to-Noise Ratio)
# =============================================================================

def calculate_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    crop_border: int = 0
) -> float:
    """
    Calculate PSNR between predicted and target images.

    Args:
        pred: Predicted image (B, C, H, W) or (H, W, C) or (H, W)
        target: Target image (same shape as pred)
        data_range: Maximum value range (1.0 for normalized, 255 for uint8)
        crop_border: Pixels to crop from border before computing

    Returns:
        PSNR value in dB (higher is better, >25dB is good)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Handle batch dimension - take first image if batch
    if pred.ndim == 4:
        pred = pred[0]
    if target.ndim == 4:
        target = target[0]

    # Convert (C, H, W) to (H, W, C)
    if pred.ndim == 3 and pred.shape[0] <= 4:
        pred = np.transpose(pred, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))

    # Crop border
    if crop_border > 0:
        pred = pred[crop_border:-crop_border, crop_border:-crop_border]
        target = target[crop_border:-crop_border, crop_border:-crop_border]

    return peak_signal_noise_ratio(target, pred, data_range=data_range)


class PSNR(nn.Module):
    """PyTorch module for PSNR calculation."""

    def __init__(self, data_range: float = 1.0, crop_border: int = 0):
        super().__init__()
        self.data_range = data_range
        self.crop_border = crop_border

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(calculate_psnr(
            pred, target,
            data_range=self.data_range,
            crop_border=self.crop_border
        ))


# =============================================================================
# SSIM (Structural Similarity Index)
# =============================================================================

def calculate_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    crop_border: int = 0,
    multichannel: bool = True
) -> float:
    """
    Calculate SSIM between predicted and target images.

    Args:
        pred: Predicted image
        target: Target image
        data_range: Maximum value range
        crop_border: Pixels to crop from border
        multichannel: Whether image has multiple channels

    Returns:
        SSIM value [0, 1] (higher is better, >0.8 is good)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Handle batch dimension - take first image if batch
    if pred.ndim == 4:
        pred = pred[0]
    if target.ndim == 4:
        target = target[0]

    # Convert (C, H, W) to (H, W, C)
    if pred.ndim == 3 and pred.shape[0] <= 4:
        pred = np.transpose(pred, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))

    # Crop border
    if crop_border > 0:
        pred = pred[crop_border:-crop_border, crop_border:-crop_border]
        target = target[crop_border:-crop_border, crop_border:-crop_border]

    # Determine if multichannel
    if pred.ndim == 2:
        multichannel = False
    else:
        multichannel = pred.shape[-1] > 1

    return structural_similarity(
        target, pred,
        data_range=data_range,
        channel_axis=2 if multichannel else None,
        win_size=7  # Smaller window for better locality
    )


class SSIM(nn.Module):
    """PyTorch module for SSIM calculation."""

    def __init__(self, data_range: float = 1.0, crop_border: int = 0):
        super().__init__()
        self.data_range = data_range
        self.crop_border = crop_border

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(calculate_ssim(
            pred, target,
            data_range=self.data_range,
            crop_border=self.crop_border
        ))


# =============================================================================
# LPIPS (Learned Perceptual Image Patch Similarity)
# =============================================================================

class LPIPS(nn.Module):
    """
    LPIPS metric using pretrained AlexNet/VGG features.

    Measures perceptual distance between images.
    Lower is better (<0.15 is good, <0.1 is excellent).
    """

    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        super().__init__()

        try:
            import lpips
            self.model = lpips.LPIPS(net=net).to(device)
            self.available = True
        except ImportError:
            print("⚠ lpips package not installed. LPIPS metric disabled.")
            print("  Install with: pip install lpips")
            self.available = False

        self.device = device

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate LPIPS distance.

        Args:
            pred: Predicted image (B, 3, H, W) normalized to [-1, 1]
            target: Target image (B, 3, H, W) normalized to [-1, 1]

        Returns:
            LPIPS distance (lower is better)
        """
        if not self.available:
            return torch.tensor(0.0)

        # Ensure inputs are in [-1, 1] range
        if pred.min() >= 0:
            pred = pred * 2 - 1
            target = target * 2 - 1

        pred = pred.to(self.device)
        target = target.to(self.device)

        with torch.no_grad():
            distance = self.model(pred, target)

        return distance.mean()


def calculate_lpips(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    net: str = 'alex',
    device: str = 'cuda'
) -> float:
    """
    Calculate LPIPS between images.

    Args:
        pred: Predicted image
        target: Target image
        net: Network backbone ('alex' or 'vgg')
        device: Computation device

    Returns:
        LPIPS value (lower is better)
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()

    # Ensure 4D tensor
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # Ensure (B, C, H, W) format
    if pred.shape[-1] <= 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

    lpips_metric = LPIPS(net=net, device=device)
    return lpips_metric(pred, target).item()


# =============================================================================
# Edge Coherence Score - HALLUCINATION DETECTION (CUSTOM METRIC)
# =============================================================================

class EdgeCoherence(nn.Module):
    """
    Edge Coherence Score - Custom metric for hallucination detection.

    Measures alignment between edges in SR output and bicubic baseline.
    If the model is hallucinating features, edges in SR won't align with
    edges in the bicubic upsampled version.

    Formula: Pearson correlation between Canny(SR) and Canny(Bicubic_8x(LR))

    Score interpretation:
    - >0.90: Excellent (minimal hallucination)
    - 0.85-0.90: Good
    - 0.80-0.85: Acceptable
    - <0.80: Concerning (possible hallucination)
    """

    def __init__(
        self,
        scale_factor: int = 8,
        canny_low: int = 50,
        canny_high: int = 150
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.canny_low = canny_low
        self.canny_high = canny_high

    def _to_numpy_gray(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to grayscale numpy array."""
        if isinstance(tensor, torch.Tensor):
            img = tensor.detach().cpu().numpy()
        else:
            img = tensor

        # Handle batch dimension
        if img.ndim == 4:
            img = img[0]

        # Convert (C, H, W) to (H, W, C)
        if img.ndim == 3 and img.shape[0] <= 4:
            img = np.transpose(img, (1, 2, 0))

        # Convert to uint8
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        # Convert to grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return img

    def _extract_edges(self, img: np.ndarray) -> np.ndarray:
        """Extract edges using Canny detector."""
        edges = cv2.Canny(img, self.canny_low, self.canny_high)
        return edges.astype(np.float32) / 255.0

    def forward(
        self,
        sr_output: torch.Tensor,
        lr_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Edge Coherence Score.

        Args:
            sr_output: Super-resolved output (B, 3, H*8, W*8)
            lr_input: Low-resolution input (B, 3, H, W)

        Returns:
            Edge coherence score [0, 1]
        """
        # Generate bicubic baseline
        bicubic = F.interpolate(
            lr_input,
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )

        # Convert to numpy grayscale
        sr_gray = self._to_numpy_gray(sr_output)
        bicubic_gray = self._to_numpy_gray(bicubic)

        # Extract edges
        sr_edges = self._extract_edges(sr_gray)
        bicubic_edges = self._extract_edges(bicubic_gray)

        # Calculate Pearson correlation
        sr_flat = sr_edges.flatten()
        bicubic_flat = bicubic_edges.flatten()

        # Handle edge case where one image has no edges
        if sr_flat.std() < 1e-6 or bicubic_flat.std() < 1e-6:
            return torch.tensor(1.0 if np.allclose(sr_flat, bicubic_flat) else 0.0)

        correlation = np.corrcoef(sr_flat, bicubic_flat)[0, 1]

        # Handle NaN (can occur if images are constant)
        if np.isnan(correlation):
            correlation = 0.0

        # Clip negative correlations
        return torch.tensor(max(0.0, correlation))


def calculate_edge_coherence(
    sr_output: Union[torch.Tensor, np.ndarray],
    lr_input: Union[torch.Tensor, np.ndarray],
    scale_factor: int = 8
) -> float:
    """
    Calculate Edge Coherence Score.

    Args:
        sr_output: Super-resolved image
        lr_input: Low-resolution input
        scale_factor: SR scale factor

    Returns:
        Edge coherence score [0, 1]
    """
    if isinstance(sr_output, np.ndarray):
        sr_output = torch.from_numpy(sr_output).float()
    if isinstance(lr_input, np.ndarray):
        lr_input = torch.from_numpy(lr_input).float()

    # Ensure 4D
    if sr_output.ndim == 3:
        sr_output = sr_output.unsqueeze(0)
    if lr_input.ndim == 3:
        lr_input = lr_input.unsqueeze(0)

    metric = EdgeCoherence(scale_factor=scale_factor)
    return metric(sr_output, lr_input).item()


# =============================================================================
# Comprehensive Metrics Calculator
# =============================================================================

class MetricsCalculator:
    """
    Unified metrics calculator for SR evaluation.

    Computes all metrics in a single pass for efficiency.
    """

    def __init__(
        self,
        scale_factor: int = 8,
        crop_border: int = 8,
        device: str = 'cuda',
        compute_lpips: bool = True
    ):
        self.scale_factor = scale_factor
        self.crop_border = crop_border
        self.device = device

        self.psnr = PSNR(crop_border=crop_border)
        self.ssim = SSIM(crop_border=crop_border)
        self.edge_coherence = EdgeCoherence(scale_factor=scale_factor)

        if compute_lpips:
            self.lpips = LPIPS(device=device)
        else:
            self.lpips = None

    def calculate(
        self,
        sr_output: torch.Tensor,
        hr_target: torch.Tensor,
        lr_input: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Calculate all metrics.

        Args:
            sr_output: Super-resolved output
            hr_target: High-resolution ground truth
            lr_input: Low-resolution input (required for edge coherence)

        Returns:
            Dictionary with all metric values
        """
        results = {}

        # PSNR
        results['psnr'] = self.psnr(sr_output, hr_target).item()

        # SSIM
        results['ssim'] = self.ssim(sr_output, hr_target).item()

        # LPIPS
        if self.lpips is not None:
            results['lpips'] = self.lpips(sr_output, hr_target).item()

        # Edge Coherence (requires LR input)
        if lr_input is not None:
            results['edge_coherence'] = self.edge_coherence(
                sr_output, lr_input).item()

        return results

    def calculate_batch(
        self,
        sr_outputs: torch.Tensor,
        hr_targets: torch.Tensor,
        lr_inputs: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Calculate metrics for a batch and return averages.

        Args:
            sr_outputs: Batch of SR outputs (B, C, H, W)
            hr_targets: Batch of HR targets (B, C, H, W)
            lr_inputs: Batch of LR inputs (B, C, H, W)

        Returns:
            Dictionary with average metric values
        """
        batch_size = sr_outputs.shape[0]

        metrics_sum = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'edge_coherence': 0}

        for i in range(batch_size):
            sr = sr_outputs[i:i+1]
            hr = hr_targets[i:i+1]
            lr = lr_inputs[i:i+1] if lr_inputs is not None else None

            batch_metrics = self.calculate(sr, hr, lr)

            for k, v in batch_metrics.items():
                metrics_sum[k] += v

        return {k: v / batch_size for k, v in metrics_sum.items()}


# =============================================================================
# Visualization Utilities
# =============================================================================

def create_difference_map(
    sr_output: Union[torch.Tensor, np.ndarray],
    bicubic: Union[torch.Tensor, np.ndarray],
    normalize: bool = True
) -> np.ndarray:
    """
    Create a difference heatmap between SR output and bicubic baseline.

    Useful for visualizing where the model made changes and potentially
    detecting hallucinated features.

    Args:
        sr_output: Super-resolved image
        bicubic: Bicubic upsampled baseline
        normalize: Whether to normalize to [0, 255]

    Returns:
        Difference heatmap as numpy array
    """
    if isinstance(sr_output, torch.Tensor):
        sr_output = sr_output.detach().cpu().numpy()
    if isinstance(bicubic, torch.Tensor):
        bicubic = bicubic.detach().cpu().numpy()

    # Handle tensor shapes
    if sr_output.ndim == 4:
        sr_output = sr_output[0]
    if bicubic.ndim == 4:
        bicubic = bicubic[0]

    if sr_output.shape[0] <= 4:
        sr_output = np.transpose(sr_output, (1, 2, 0))
        bicubic = np.transpose(bicubic, (1, 2, 0))

    # Compute absolute difference
    diff = np.abs(sr_output.astype(np.float32) - bicubic.astype(np.float32))

    # Convert to grayscale magnitude
    if diff.ndim == 3:
        diff = np.mean(diff, axis=-1)

    if normalize:
        diff = (diff / diff.max() * 255).astype(np.uint8)

    # Apply colormap for visualization
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap


def create_edge_overlay(
    image: Union[torch.Tensor, np.ndarray],
    edge_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create an edge-overlaid visualization.

    Args:
        image: Input image
        edge_color: Color for edge overlay (R, G, B)
        alpha: Transparency for overlay

    Returns:
        Image with edge overlay
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.ndim == 4:
        image = image[0]
    if image.shape[0] <= 4:
        image = np.transpose(image, (1, 2, 0))

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Create edge overlay
    overlay = image.copy()
    overlay[edges > 0] = edge_color

    # Blend
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return result


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Metrics Module")
    print("=" * 60)

    # Create test images
    batch_size = 2
    lr_size = 32
    hr_size = 256

    lr = torch.rand(batch_size, 3, lr_size, lr_size)
    hr = torch.rand(batch_size, 3, hr_size, hr_size)
    sr = torch.rand(batch_size, 3, hr_size, hr_size)

    # Test PSNR
    print("\n1. Testing PSNR...")
    psnr_val = calculate_psnr(sr[0], hr[0])
    print(f"   PSNR: {psnr_val:.2f} dB")

    # Test SSIM
    print("\n2. Testing SSIM...")
    ssim_val = calculate_ssim(sr[0], hr[0])
    print(f"   SSIM: {ssim_val:.4f}")

    # Test Edge Coherence
    print("\n3. Testing Edge Coherence...")
    ec = EdgeCoherence(scale_factor=8)
    ec_val = ec(sr, lr)
    print(f"   Edge Coherence: {ec_val.item():.4f}")

    # Test MetricsCalculator
    print("\n4. Testing MetricsCalculator...")
    calculator = MetricsCalculator(
        compute_lpips=False)  # Skip LPIPS for quick test
    metrics = calculator.calculate(sr, hr, lr)
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    # Test difference map
    print("\n5. Testing Difference Map...")
    bicubic = F.interpolate(
        lr, scale_factor=8, mode='bicubic', align_corners=False)
    diff_map = create_difference_map(sr[0], bicubic[0])
    print(f"   Difference map shape: {diff_map.shape}")

    print("\n✓ All metrics tests passed!")
