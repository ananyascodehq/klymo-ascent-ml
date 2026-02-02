"""
Inference Pipeline for Geospatial Super-Resolution
====================================================

Features:
- Tile-based processing for memory efficiency
- Batch inference support
- Overlap-tile strategy for seamless reconstruction
- Support for various input formats (numpy, PIL, torch)
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional, Tuple, List
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import local modules
from model import create_generator, ESRGANGenerator, ESRGANGeneratorRGB
from metrics import MetricsCalculator, create_difference_map


# =============================================================================
# Inference Engine
# =============================================================================

class SuperResolutionInference:
    """
    Inference engine for geospatial super-resolution.

    Handles:
    - Model loading from checkpoints
    - Tile-based processing for large images
    - Overlap strategy for seamless boundaries
    - Various input/output formats
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        scale_factor: int = 8,
        tile_size: int = 128,
        tile_overlap: int = 16,
        use_indices: bool = False
    ):
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            device: Computation device ('cuda' or 'cpu')
            scale_factor: Super-resolution scale factor
            tile_size: Size of processing tiles (LR resolution)
            tile_overlap: Overlap between tiles to avoid boundary artifacts
            use_indices: Whether model uses 16-channel input with indices
        """
        self.device = device
        self.scale_factor = scale_factor
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.use_indices = use_indices

        # Create model
        self.model = create_generator(
            use_gam=use_indices,
            in_channels=16 if use_indices else 3,
            scale_factor=scale_factor
        )

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.model = self.model.to(device)
        self.model.eval()

        # Metrics calculator
        self.metrics = MetricsCalculator(
            scale_factor=scale_factor,
            compute_lpips=False
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'generator' in checkpoint:
            self.model.load_state_dict(checkpoint['generator'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"✓ Loaded checkpoint: {checkpoint_path}")

        if 'psnr' in checkpoint:
            print(f"  Checkpoint PSNR: {checkpoint['psnr']:.2f}dB")
        if 'ssim' in checkpoint:
            print(f"  Checkpoint SSIM: {checkpoint['ssim']:.4f}")

    def preprocess(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess input image to tensor.

        Args:
            image: Input image (numpy, PIL, or tensor)

        Returns:
            Preprocessed tensor (1, C, H, W) normalized to [0, 1]
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if isinstance(image, np.ndarray):
            # Handle grayscale
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)

            # Convert (H, W, C) to (C, H, W)
            if image.shape[-1] <= 4:
                image = np.transpose(image, (2, 0, 1))

            # Normalize to [0, 1]
            if image.max() > 1:
                image = image.astype(np.float32) / 255.0

            image = torch.from_numpy(image).float()

        # Ensure 4D
        if image.ndim == 3:
            image = image.unsqueeze(0)

        return image.to(self.device)

    def postprocess(
        self,
        tensor: torch.Tensor,
        output_type: str = 'numpy'
    ) -> Union[np.ndarray, Image.Image, torch.Tensor]:
        """
        Postprocess output tensor to desired format.

        Args:
            tensor: Output tensor (B, C, H, W)
            output_type: 'numpy', 'pil', or 'tensor'

        Returns:
            Postprocessed image
        """
        # Clamp to valid range
        tensor = torch.clamp(tensor, 0, 1)

        if output_type == 'tensor':
            return tensor

        # Convert to numpy
        image = tensor.squeeze(0).cpu().numpy()

        # Convert (C, H, W) to (H, W, C)
        if image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))

        # Scale to uint8
        image = (image * 255).astype(np.uint8)

        if output_type == 'pil':
            return Image.fromarray(image)

        return image

    @torch.no_grad()
    def enhance_single(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        output_type: str = 'numpy'
    ) -> Union[np.ndarray, Image.Image, torch.Tensor]:
        """
        Enhance a single image (small enough to fit in memory).

        Args:
            image: Input LR image
            output_type: Output format ('numpy', 'pil', 'tensor')

        Returns:
            Super-resolved image
        """
        # Preprocess
        lr = self.preprocess(image)

        # Forward pass
        sr = self.model(lr)

        # Postprocess
        return self.postprocess(sr, output_type)

    @torch.no_grad()
    def enhance_tiled(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        output_type: str = 'numpy',
        show_progress: bool = True
    ) -> Union[np.ndarray, Image.Image, torch.Tensor]:
        """
        Enhance a large image using tile-based processing.

        Uses overlap-tile strategy to avoid boundary artifacts.

        Args:
            image: Input LR image (can be large)
            output_type: Output format
            show_progress: Show progress bar

        Returns:
            Super-resolved image
        """
        # Preprocess
        lr = self.preprocess(image)
        _, c, h, w = lr.shape

        # Calculate output size
        hr_h = h * self.scale_factor
        hr_w = w * self.scale_factor

        # Initialize output with overlap accumulation
        sr_output = torch.zeros(1, 3, hr_h, hr_w, device=self.device)
        weight_map = torch.zeros(1, 1, hr_h, hr_w, device=self.device)

        # Calculate tile grid
        step = self.tile_size - self.tile_overlap
        hr_step = step * self.scale_factor
        hr_tile_size = self.tile_size * self.scale_factor
        hr_overlap = self.tile_overlap * self.scale_factor

        # Create tiles
        tiles = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Handle boundary tiles
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                y_start = max(0, y_end - self.tile_size)
                x_start = max(0, x_end - self.tile_size)

                tiles.append((y_start, x_start, y_end, x_end))

        # Process tiles
        iterator = tqdm(
            tiles, desc="Processing tiles") if show_progress else tiles

        for y_start, x_start, y_end, x_end in iterator:
            # Extract tile
            tile = lr[:, :, y_start:y_end, x_start:x_end]

            # Pad if needed
            pad_h = self.tile_size - tile.shape[2]
            pad_w = self.tile_size - tile.shape[3]
            if pad_h > 0 or pad_w > 0:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')

            # Process tile
            sr_tile = self.model(tile)

            # Remove padding
            if pad_h > 0:
                sr_tile = sr_tile[:, :, :hr_tile_size -
                                  pad_h * self.scale_factor, :]
            if pad_w > 0:
                sr_tile = sr_tile[:, :, :, :hr_tile_size -
                                  pad_w * self.scale_factor]

            # Calculate HR positions
            hr_y_start = y_start * self.scale_factor
            hr_x_start = x_start * self.scale_factor
            hr_y_end = hr_y_start + sr_tile.shape[2]
            hr_x_end = hr_x_start + sr_tile.shape[3]

            # Create blending weight (higher in center, lower at edges)
            tile_weight = self._create_blend_weight(
                sr_tile.shape[2], sr_tile.shape[3], hr_overlap
            ).to(self.device)

            # Accumulate
            sr_output[:, :, hr_y_start:hr_y_end,
                      hr_x_start:hr_x_end] += sr_tile * tile_weight
            weight_map[:, :, hr_y_start:hr_y_end,
                       hr_x_start:hr_x_end] += tile_weight

        # Normalize by weight
        sr_output = sr_output / (weight_map + 1e-8)

        return self.postprocess(sr_output, output_type)

    def _create_blend_weight(self, h: int, w: int, overlap: int) -> torch.Tensor:
        """Create blending weight map for tile merging."""
        weight = torch.ones(1, 1, h, w)

        if overlap > 0:
            # Create linear ramps for edges
            ramp = torch.linspace(0, 1, overlap)

            # Top edge
            weight[:, :, :overlap, :] *= ramp.view(-1, 1)
            # Bottom edge
            weight[:, :, -overlap:, :] *= ramp.flip(0).view(-1, 1)
            # Left edge
            weight[:, :, :, :overlap] *= ramp.view(1, -1)
            # Right edge
            weight[:, :, :, -overlap:] *= ramp.flip(0).view(1, -1)

        return weight

    def enhance(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        output_type: str = 'numpy',
        auto_tile: bool = True,
        tile_threshold: int = 256
    ) -> Union[np.ndarray, Image.Image, torch.Tensor]:
        """
        Smart enhancement that chooses single vs tiled processing.

        Args:
            image: Input LR image
            output_type: Output format
            auto_tile: Automatically use tiling for large images
            tile_threshold: Size threshold for tiling (in pixels)

        Returns:
            Super-resolved image
        """
        # Get image size
        if isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = image.shape[-2:]

        # Decide processing method
        if auto_tile and (h > tile_threshold or w > tile_threshold):
            return self.enhance_tiled(image, output_type)
        else:
            return self.enhance_single(image, output_type)

    def evaluate(
        self,
        lr_image: Union[np.ndarray, Image.Image, torch.Tensor],
        hr_image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], dict]:
        """
        Enhance and evaluate against ground truth.

        Args:
            lr_image: Low-resolution input
            hr_image: High-resolution ground truth

        Returns:
            (sr_output, metrics_dict)
        """
        # Preprocess
        lr = self.preprocess(lr_image)
        hr = self.preprocess(hr_image)

        # Enhance
        with torch.no_grad():
            sr = self.model(lr)

        # Calculate metrics
        metrics = self.metrics.calculate(sr, hr, lr)

        return self.postprocess(sr, 'numpy'), metrics

    def compare(
        self,
        lr_image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> dict:
        """
        Generate comparison outputs: bicubic, SR, and difference map.

        Args:
            lr_image: Low-resolution input

        Returns:
            Dictionary with 'lr', 'bicubic', 'sr', 'difference' images
        """
        lr = self.preprocess(lr_image)

        # Generate outputs
        with torch.no_grad():
            sr = self.model(lr)

        bicubic = F.interpolate(
            lr, scale_factor=self.scale_factor,
            mode='bicubic', align_corners=False
        )

        # Create difference map
        diff_map = create_difference_map(sr, bicubic)

        return {
            'lr': self.postprocess(lr, 'numpy'),
            'bicubic': self.postprocess(bicubic, 'numpy'),
            'sr': self.postprocess(sr, 'numpy'),
            'difference': diff_map
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def load_model(
    checkpoint_path: str,
    device: str = 'cuda',
    scale_factor: int = 8
) -> SuperResolutionInference:
    """
    Convenience function to load a trained model.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use
        scale_factor: SR scale factor

    Returns:
        Initialized inference engine
    """
    return SuperResolutionInference(
        checkpoint_path=checkpoint_path,
        device=device,
        scale_factor=scale_factor
    )


def enhance_image(
    image_path: str,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Enhance a single image file.

    Args:
        image_path: Path to input image
        checkpoint_path: Path to model checkpoint
        output_path: Path to save output (optional)
        device: Device to use

    Returns:
        Enhanced image as numpy array
    """
    # Load image
    image = Image.open(image_path)

    # Load model and enhance
    engine = SuperResolutionInference(
        checkpoint_path=checkpoint_path,
        device=device
    )

    sr = engine.enhance(image, output_type='pil')

    # Save if output path provided
    if output_path:
        sr.save(output_path)
        print(f"✓ Saved: {output_path}")

    return np.array(sr)


def batch_enhance(
    input_dir: str,
    output_dir: str,
    checkpoint_path: str,
    device: str = 'cuda',
    extensions: List[str] = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
) -> List[str]:
    """
    Enhance all images in a directory.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        checkpoint_path: Path to model checkpoint
        device: Device to use
        extensions: Valid file extensions

    Returns:
        List of output file paths
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"No images found in {input_dir}")
        return []

    # Load model
    engine = SuperResolutionInference(
        checkpoint_path=checkpoint_path,
        device=device
    )

    # Process images
    output_files = []
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            image = Image.open(img_path)
            sr = engine.enhance(image, output_type='pil')

            out_path = output_path / f"{img_path.stem}_sr{img_path.suffix}"
            sr.save(out_path)
            output_files.append(str(out_path))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"\n✓ Processed {len(output_files)} images")
    return output_files


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='KLYMO ASCENT ML - Super-Resolution Inference'
    )
    parser.add_argument(
        'input', type=str,
        help='Input image path or directory'
    )
    parser.add_argument(
        '--checkpoint', '-c', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output path (file or directory)'
    )
    parser.add_argument(
        '--device', '-d', type=str, default='cuda',
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--tile-size', type=int, default=128,
        help='Tile size for large image processing'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Generate comparison outputs (LR, Bicubic, SR, Diff)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        # Batch processing
        output_dir = args.output or str(input_path / 'sr_output')
        batch_enhance(
            str(input_path),
            output_dir,
            args.checkpoint,
            args.device
        )
    else:
        # Single image
        output_path = args.output
        if not output_path:
            output_path = str(input_path.parent /
                              f"{input_path.stem}_sr{input_path.suffix}")

        engine = SuperResolutionInference(
            checkpoint_path=args.checkpoint,
            device=args.device,
            tile_size=args.tile_size
        )

        if args.compare:
            image = Image.open(input_path)
            results = engine.compare(image)

            # Save all outputs
            base = Path(output_path).stem
            out_dir = Path(output_path).parent

            Image.fromarray(results['bicubic']).save(
                out_dir / f"{base}_bicubic.png")
            Image.fromarray(results['sr']).save(out_dir / f"{base}_sr.png")
            Image.fromarray(results['difference']).save(
                out_dir / f"{base}_diff.png")

            print(f"✓ Saved comparison outputs to {out_dir}")
        else:
            enhance_image(str(input_path), args.checkpoint,
                          output_path, args.device)


if __name__ == "__main__":
    main()
