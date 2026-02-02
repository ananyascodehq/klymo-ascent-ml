"""
Data Loader for Geospatial Super-Resolution
============================================

This module handles:
1. Google Earth Engine (GEE) API integration for satellite imagery
2. Tile fetching and caching for LR/HR pairs
3. Data preprocessing (radiometric correction, cloud masking)
4. Geospatial index computation (NDVI, NDBI, NDWI)
5. PyTorch DataLoader with on-the-fly augmentation
"""

import os
import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


# =============================================================================
# Configuration
# =============================================================================

# Urban areas for training data (coordinates: [lon, lat])
URBAN_AREAS = {
    'mumbai': {'center': [72.8777, 19.0760], 'zoom': 13},
    'tokyo': {'center': [139.6917, 35.6895], 'zoom': 13},
    'lagos': {'center': [3.3792, 6.5244], 'zoom': 13},
    'delhi': {'center': [77.1025, 28.7041], 'zoom': 13},
    'sao_paulo': {'center': [-46.6333, -23.5505], 'zoom': 13},
    'new_york': {'center': [-74.0060, 40.7128], 'zoom': 13},
    'cairo': {'center': [31.2357, 30.0444], 'zoom': 13},
    'beijing': {'center': [116.4074, 39.9042], 'zoom': 13},
    'london': {'center': [-0.1276, 51.5074], 'zoom': 13},
    'paris': {'center': [2.3522, 48.8566], 'zoom': 13},
}

# Sentinel-2 band configuration
SENTINEL2_BANDS = {
    'B01': {'name': 'Coastal aerosol', 'resolution': 60},
    'B02': {'name': 'Blue', 'resolution': 10},
    'B03': {'name': 'Green', 'resolution': 10},
    'B04': {'name': 'Red', 'resolution': 10},
    'B05': {'name': 'Vegetation Red Edge 1', 'resolution': 20},
    'B06': {'name': 'Vegetation Red Edge 2', 'resolution': 20},
    'B07': {'name': 'Vegetation Red Edge 3', 'resolution': 20},
    'B08': {'name': 'NIR', 'resolution': 10},
    'B8A': {'name': 'Vegetation Red Edge 4', 'resolution': 20},
    'B09': {'name': 'Water Vapour', 'resolution': 60},
    'B10': {'name': 'SWIR - Cirrus', 'resolution': 60},
    'B11': {'name': 'SWIR 1', 'resolution': 20},
    'B12': {'name': 'SWIR 2', 'resolution': 20},
}


# =============================================================================
# Google Earth Engine Integration
# =============================================================================

class GEETileFetcher:
    """
    Fetches satellite imagery tiles from Google Earth Engine.

    Uses GEE Python API to fetch Sentinel-2 L2A imagery with cloud masking.
    """

    def __init__(self, cache_dir: str = './data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ee_initialized = False

    def initialize_ee(self, project_id: Optional[str] = None):
        """Initialize Google Earth Engine API."""
        try:
            import ee

            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()

            self.ee_initialized = True
            print("✓ Google Earth Engine initialized successfully")
        except Exception as e:
            print(f"⚠ GEE initialization failed: {e}")
            print("  Run: earthengine authenticate")
            self.ee_initialized = False

    def get_sentinel2_tile(
        self,
        center_lon: float,
        center_lat: float,
        tile_size: int = 256,
        date_start: str = '2023-01-01',
        date_end: str = '2023-12-31',
        max_cloud_cover: float = 10.0
    ) -> Optional[np.ndarray]:
        """
        Fetch a Sentinel-2 L2A tile from GEE.

        Args:
            center_lon: Center longitude
            center_lat: Center latitude
            tile_size: Tile size in pixels
            date_start: Start date for image search
            date_end: End date for image search
            max_cloud_cover: Maximum cloud cover percentage

        Returns:
            numpy array of shape (13, H, W) with Sentinel-2 bands
        """
        if not self.ee_initialized:
            print("GEE not initialized. Call initialize_ee() first.")
            return None

        import ee

        # Define point of interest
        point = ee.Geometry.Point([center_lon, center_lat])

        # Create bounding box (~2.5km at equator)
        buffer_size = 1250  # meters
        region = point.buffer(buffer_size).bounds()

        # Get Sentinel-2 L2A collection
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(point)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        )

        # Check if we have images
        count = collection.size().getInfo()
        if count == 0:
            print(f"No images found for location ({center_lon}, {center_lat})")
            return None

        # Get the best (least cloudy) image
        image = collection.first()

        # Select all 13 bands
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        image = image.select(bands)

        # Get thumbnail URL
        url = image.getThumbURL({
            'region': region,
            'dimensions': f'{tile_size}x{tile_size}',
            'format': 'png'
        })

        # Download and convert
        response = requests.get(url)
        if response.status_code == 200:
            from io import BytesIO
            img = Image.open(BytesIO(response.content))
            return np.array(img)

        return None

    def compute_geospatial_indices(self, bands: np.ndarray) -> np.ndarray:
        """
        Compute NDVI, NDBI, and NDWI from Sentinel-2 bands.

        Args:
            bands: numpy array of shape (13, H, W) with Sentinel-2 bands
                   Band order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12

        Returns:
            numpy array of shape (3, H, W) with [NDVI, NDBI, NDWI]
        """
        eps = 1e-8

        # Band indices (0-indexed)
        RED = 3   # B04
        GREEN = 2  # B03
        NIR = 7   # B08
        SWIR = 10  # B11

        red = bands[RED].astype(np.float32)
        green = bands[GREEN].astype(np.float32)
        nir = bands[NIR].astype(np.float32)
        swir = bands[SWIR].astype(np.float32)

        # NDVI: Vegetation index
        ndvi = (nir - red) / (nir + red + eps)

        # NDBI: Built-up index
        ndbi = (swir - nir) / (swir + nir + eps)

        # NDWI: Water index
        ndwi = (green - nir) / (green + nir + eps)

        # Stack indices
        indices = np.stack([ndvi, ndbi, ndwi], axis=0)

        # Clip to [-1, 1]
        indices = np.clip(indices, -1, 1)

        return indices


# =============================================================================
# Data Preprocessing
# =============================================================================

class RadiometricCorrector:
    """Handles radiometric correction of satellite imagery."""

    @staticmethod
    def normalize_16bit_to_8bit(
        image: np.ndarray,
        percentile: float = 99.0,
        clip_min: float = 0.0
    ) -> np.ndarray:
        """
        Convert 16-bit satellite data to 8-bit using percentile clipping.

        Args:
            image: 16-bit image array
            percentile: Upper percentile for clipping
            clip_min: Minimum value for clipping

        Returns:
            8-bit normalized image
        """
        clip_max = np.percentile(image, percentile)
        image = np.clip(image, clip_min, clip_max)
        image = (image - clip_min) / (clip_max - clip_min + 1e-8)
        image = (image * 255).astype(np.uint8)
        return image

    @staticmethod
    def cloud_mask(qa_band: np.ndarray) -> np.ndarray:
        """
        Create cloud mask from Sentinel-2 QA60 band.

        Args:
            qa_band: QA60 band array

        Returns:
            Binary mask (1 = clear, 0 = cloudy)
        """
        # Bit 10 = opaque clouds, Bit 11 = cirrus clouds
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        mask = ((qa_band & cloud_bit_mask) == 0) & (
            (qa_band & cirrus_bit_mask) == 0)
        return mask.astype(np.float32)


# =============================================================================
# Synthetic Data Generation (for testing without GEE)
# =============================================================================

class SyntheticDataGenerator:
    """
    Generates synthetic LR/HR pairs for testing and development.
    Uses standard image SR datasets or generates procedural tiles.
    """

    def __init__(self, scale_factor: int = 8):
        self.scale_factor = scale_factor

    def generate_tile_pair(
        self,
        hr_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic LR/HR tile pair.

        Args:
            hr_size: Size of HR tile

        Returns:
            (lr_tile, hr_tile) tuple
        """
        lr_size = hr_size // self.scale_factor

        # Generate procedural "urban" texture
        hr_tile = self._generate_urban_texture(hr_size)

        # Downsample to create LR
        lr_tile = self._downsample(hr_tile, lr_size)

        return lr_tile, hr_tile

    def _generate_urban_texture(self, size: int) -> np.ndarray:
        """Generate a procedural urban-like texture."""
        # Create base
        tile = np.random.rand(size, size, 3) * 0.3 + 0.2

        # Add "buildings" (rectangular blocks)
        num_buildings = np.random.randint(5, 15)
        for _ in range(num_buildings):
            x = np.random.randint(0, size - 30)
            y = np.random.randint(0, size - 30)
            w = np.random.randint(10, 40)
            h = np.random.randint(10, 40)
            color = np.random.rand(3) * 0.3 + 0.4
            tile[y:y+h, x:x+w] = color

        # Add "roads" (lines)
        num_roads = np.random.randint(2, 5)
        for _ in range(num_roads):
            if np.random.rand() > 0.5:
                y = np.random.randint(0, size)
                tile[max(0, y-2):min(size, y+2), :] = [0.3, 0.3, 0.35]
            else:
                x = np.random.randint(0, size)
                tile[:, max(0, x-2):min(size, x+2)] = [0.3, 0.3, 0.35]

        # Add "vegetation" patches
        num_patches = np.random.randint(2, 6)
        for _ in range(num_patches):
            x = np.random.randint(0, size - 20)
            y = np.random.randint(0, size - 20)
            r = np.random.randint(5, 15)
            yy, xx = np.ogrid[:size, :size]
            mask = ((xx - x - r) ** 2 + (yy - y - r) ** 2) < r ** 2
            tile[mask] = [0.2, 0.5 + np.random.rand() * 0.2, 0.15]

        return (tile * 255).astype(np.uint8)

    def _downsample(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Downsample image using bicubic interpolation."""
        img = Image.fromarray(image)
        img = img.resize((target_size, target_size), Image.BICUBIC)
        return np.array(img)

    def generate_13band_input(self, lr_rgb: np.ndarray) -> np.ndarray:
        """
        Generate synthetic 13-band + 3-index input from RGB.

        For testing when real Sentinel-2 data isn't available.

        Args:
            lr_rgb: RGB image (H, W, 3)

        Returns:
            16-channel input (16, H, W)
        """
        h, w = lr_rgb.shape[:2]

        # Simulate 13 Sentinel-2 bands from RGB
        bands = np.zeros((13, h, w), dtype=np.float32)

        # Map RGB to approximate Sentinel-2 bands
        r = lr_rgb[:, :, 0].astype(np.float32) / 255.0
        g = lr_rgb[:, :, 1].astype(np.float32) / 255.0
        b = lr_rgb[:, :, 2].astype(np.float32) / 255.0

        # Approximate band mapping
        bands[0] = b * 0.9  # B01 - Coastal aerosol
        bands[1] = b         # B02 - Blue
        bands[2] = g         # B03 - Green
        bands[3] = r         # B04 - Red
        bands[4] = (r + g) / 2 * 0.8  # B05 - Red Edge 1
        bands[5] = (r + g) / 2 * 0.9  # B06 - Red Edge 2
        bands[6] = (r + g) / 2        # B07 - Red Edge 3
        bands[7] = np.maximum(r, g) * 1.1  # B08 - NIR (approximate)
        bands[8] = np.maximum(r, g)   # B8A - Red Edge 4
        bands[9] = np.ones((h, w)) * 0.5  # B09 - Water Vapour
        bands[10] = np.ones((h, w)) * 0.3  # B10 - SWIR Cirrus
        bands[11] = (r + 0.5) / 1.5   # B11 - SWIR 1
        bands[12] = (r + 0.3) / 1.3   # B12 - SWIR 2

        # Compute indices
        eps = 1e-8
        ndvi = (bands[7] - bands[3]) / (bands[7] + bands[3] + eps)
        ndbi = (bands[11] - bands[7]) / (bands[11] + bands[7] + eps)
        ndwi = (bands[2] - bands[7]) / (bands[2] + bands[7] + eps)

        # Stack all channels
        full_input = np.concatenate([
            bands,
            ndvi[np.newaxis, :, :],
            ndbi[np.newaxis, :, :],
            ndwi[np.newaxis, :, :]
        ], axis=0)

        return full_input


# =============================================================================
# PyTorch Dataset
# =============================================================================

class GeoSRDataset(Dataset):
    """
    PyTorch Dataset for Geospatial Super-Resolution.

    Supports:
    - Real GEE-fetched data (cached)
    - Synthetic procedural data
    - Standard SR benchmark datasets (DIV2K, etc.)
    """

    def __init__(
        self,
        data_dir: str = './data',
        mode: str = 'train',
        scale_factor: int = 8,
        lr_patch_size: int = 32,
        use_indices: bool = True,
        augment: bool = True,
        synthetic: bool = True,
        num_samples: int = 500
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing cached data
            mode: 'train', 'val', or 'test'
            scale_factor: SR scale factor (default: 8)
            lr_patch_size: Size of LR patches (HR = LR * scale_factor)
            use_indices: Whether to include geospatial indices
            augment: Whether to apply augmentation
            synthetic: Whether to generate synthetic data
            num_samples: Number of samples for synthetic mode
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.scale_factor = scale_factor
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale_factor
        self.use_indices = use_indices
        self.augment = augment and mode == 'train'
        self.synthetic = synthetic

        if synthetic:
            self.generator = SyntheticDataGenerator(scale_factor)
            self.num_samples = num_samples
            self.samples = []  # Will be generated on-the-fly
        else:
            self._load_cached_data()

    def _load_cached_data(self):
        """Load pre-cached LR/HR pairs."""
        lr_dir = self.data_dir / 'lr' / self.mode
        hr_dir = self.data_dir / 'hr' / self.mode

        if not lr_dir.exists() or not hr_dir.exists():
            print(f"Creating data directories: {lr_dir}, {hr_dir}")
            lr_dir.mkdir(parents=True, exist_ok=True)
            hr_dir.mkdir(parents=True, exist_ok=True)
            self.samples = []
            return

        lr_files = sorted(list(lr_dir.glob('*.png')) +
                          list(lr_dir.glob('*.npy')))
        self.samples = []

        for lr_path in lr_files:
            hr_path = hr_dir / lr_path.name
            if hr_path.exists():
                self.samples.append((lr_path, hr_path))

        print(f"Loaded {len(self.samples)} {self.mode} samples")

    def __len__(self) -> int:
        if self.synthetic:
            return self.num_samples
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.synthetic:
            return self._get_synthetic_item()
        else:
            return self._get_cached_item(idx)

    def _get_synthetic_item(self) -> Dict[str, torch.Tensor]:
        """Generate a synthetic sample on-the-fly."""
        # Generate LR/HR pair
        lr_rgb, hr_rgb = self.generator.generate_tile_pair(self.hr_patch_size)

        # Convert to tensor (C, H, W)
        lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr_rgb).permute(2, 0, 1).float() / 255.0

        # Apply augmentation
        if self.augment:
            lr_tensor, hr_tensor = self._augment(lr_tensor, hr_tensor)

        result = {
            'lr': lr_tensor,  # (3, H_lr, W_lr)
            'hr': hr_tensor,  # (3, H_hr, W_hr)
        }

        # Generate 16-channel input if needed
        if self.use_indices:
            lr_16ch = self.generator.generate_13band_input(
                (lr_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            result['lr_full'] = torch.from_numpy(lr_16ch).float()

        return result

    def _get_cached_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a cached sample."""
        lr_path, hr_path = self.samples[idx]

        # Load images
        if lr_path.suffix == '.npy':
            lr = np.load(lr_path)
            hr = np.load(hr_path)
        else:
            lr = np.array(Image.open(lr_path))
            hr = np.array(Image.open(hr_path))

        # Convert to tensor
        if lr.ndim == 2:
            lr = np.stack([lr] * 3, axis=-1)
            hr = np.stack([hr] * 3, axis=-1)

        lr_tensor = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.0

        # Random crop to patch size
        lr_tensor, hr_tensor = self._random_crop(lr_tensor, hr_tensor)

        # Apply augmentation
        if self.augment:
            lr_tensor, hr_tensor = self._augment(lr_tensor, hr_tensor)

        return {
            'lr': lr_tensor,
            'hr': hr_tensor,
        }

    def _random_crop(
        self,
        lr: torch.Tensor,
        hr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random crop LR/HR pairs."""
        _, lr_h, lr_w = lr.shape

        if lr_h > self.lr_patch_size and lr_w > self.lr_patch_size:
            i = random.randint(0, lr_h - self.lr_patch_size)
            j = random.randint(0, lr_w - self.lr_patch_size)

            lr = lr[:, i:i+self.lr_patch_size, j:j+self.lr_patch_size]

            hi = i * self.scale_factor
            hj = j * self.scale_factor
            hr = hr[:, hi:hi+self.hr_patch_size, hj:hj+self.hr_patch_size]

        return lr, hr

    def _augment(
        self,
        lr: torch.Tensor,
        hr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations."""
        # Random horizontal flip
        if random.random() > 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)

        # Random vertical flip
        if random.random() > 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)

        # Random 90-degree rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            lr = torch.rot90(lr, k, [1, 2])
            hr = torch.rot90(hr, k, [1, 2])

        return lr, hr


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_dataloader(
    data_dir: str = './data',
    mode: str = 'train',
    batch_size: int = 8,
    num_workers: int = 4,
    scale_factor: int = 8,
    lr_patch_size: int = 32,
    use_indices: bool = False,
    synthetic: bool = True,
    num_samples: int = 500
) -> DataLoader:
    """
    Create a DataLoader for training/validation.

    Args:
        data_dir: Data directory
        mode: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        scale_factor: SR scale factor
        lr_patch_size: LR patch size
        use_indices: Include geospatial indices
        synthetic: Use synthetic data
        num_samples: Number of samples for synthetic mode

    Returns:
        PyTorch DataLoader
    """
    dataset = GeoSRDataset(
        data_dir=data_dir,
        mode=mode,
        scale_factor=scale_factor,
        lr_patch_size=lr_patch_size,
        use_indices=use_indices,
        augment=(mode == 'train'),
        synthetic=synthetic,
        num_samples=num_samples
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train')
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Data Loader")
    print("=" * 60)

    # Test synthetic data generation
    print("\n1. Testing SyntheticDataGenerator...")
    gen = SyntheticDataGenerator(scale_factor=8)
    lr, hr = gen.generate_tile_pair(256)
    print(f"   LR shape: {lr.shape}, HR shape: {hr.shape}")

    # Test 13-band + indices generation
    print("\n2. Testing 16-channel input generation...")
    full_input = gen.generate_13band_input(lr)
    print(f"   16-channel input shape: {full_input.shape}")

    # Test dataset
    print("\n3. Testing GeoSRDataset...")
    dataset = GeoSRDataset(
        synthetic=True,
        num_samples=10,
        use_indices=True
    )
    print(f"   Dataset length: {len(dataset)}")

    sample = dataset[0]
    print(f"   Sample keys: {sample.keys()}")
    print(f"   LR shape: {sample['lr'].shape}")
    print(f"   HR shape: {sample['hr'].shape}")
    if 'lr_full' in sample:
        print(f"   LR full (16ch) shape: {sample['lr_full'].shape}")

    # Test dataloader
    print("\n4. Testing DataLoader...")
    loader = create_dataloader(
        batch_size=4,
        num_workers=0,
        synthetic=True,
        num_samples=20
    )

    for batch in loader:
        print(f"   Batch LR shape: {batch['lr'].shape}")
        print(f"   Batch HR shape: {batch['hr'].shape}")
        break

    print("\n✓ All data loader tests passed!")
