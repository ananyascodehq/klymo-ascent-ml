"""
Demo Utilities for Geospatial Super-Resolution
================================================

Helper functions for image processing and visualization
in the Streamlit demo application.
"""

import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Union
import io
import base64


# =============================================================================
# Image Loading and Saving
# =============================================================================

def load_image(
    source: Union[str, bytes, io.BytesIO],
    max_size: Optional[int] = None
) -> np.ndarray:
    """
    Load image from various sources.

    Args:
        source: File path, bytes, or BytesIO object
        max_size: Maximum dimension (will resize if larger)

    Returns:
        RGB numpy array
    """
    if isinstance(source, str):
        image = Image.open(source)
    elif isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(source)

    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize if needed
    if max_size and (image.width > max_size or image.height > max_size):
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    return np.array(image)


def save_image(image: np.ndarray, path: str, quality: int = 95):
    """
    Save image to file.

    Args:
        image: RGB numpy array
        path: Output file path
        quality: JPEG quality (1-100)
    """
    img = Image.fromarray(image)

    if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
        img.save(path, 'JPEG', quality=quality)
    else:
        img.save(path)


def image_to_base64(image: np.ndarray, format: str = 'PNG') -> str:
    """
    Convert image to base64 string for web display.

    Args:
        image: RGB numpy array
        format: Image format ('PNG', 'JPEG')

    Returns:
        Base64 encoded string
    """
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()


# =============================================================================
# Image Processing
# =============================================================================

def resize_image(
    image: np.ndarray,
    scale: float = None,
    size: Tuple[int, int] = None,
    method: str = 'bicubic'
) -> np.ndarray:
    """
    Resize image.

    Args:
        image: Input image
        scale: Scale factor (alternative to size)
        size: Target (width, height)
        method: 'nearest', 'bilinear', 'bicubic', 'lanczos'

    Returns:
        Resized image
    """
    methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    if scale is not None:
        size = (int(image.shape[1] * scale), int(image.shape[0] * scale))

    return cv2.resize(image, size, interpolation=methods.get(method, cv2.INTER_CUBIC))


def crop_center(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop center of image.

    Args:
        image: Input image
        crop_size: (width, height) to crop

    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    ch, cw = crop_size

    start_x = max(0, (w - cw) // 2)
    start_y = max(0, (h - ch) // 2)

    return image[start_y:start_y+ch, start_x:start_x+cw]


def pad_to_multiple(image: np.ndarray, multiple: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image so dimensions are multiples of given value.

    Args:
        image: Input image
        multiple: Dimension multiple

    Returns:
        (padded_image, (pad_h, pad_w))
    """
    h, w = image.shape[:2]

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        image = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)) if image.ndim == 3 else (
                (0, pad_h), (0, pad_w)),
            mode='reflect'
        )

    return image, (pad_h, pad_w)


# =============================================================================
# Visualization
# =============================================================================

def create_side_by_side(
    image1: np.ndarray,
    image2: np.ndarray,
    labels: Tuple[str, str] = None,
    gap: int = 10
) -> np.ndarray:
    """
    Create side-by-side comparison image.

    Args:
        image1: Left image
        image2: Right image
        labels: Optional labels (left, right)
        gap: Gap between images in pixels

    Returns:
        Combined image
    """
    # Resize to same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    if h1 != h2:
        scale = h1 / h2
        image2 = resize_image(image2, size=(int(w2 * scale), h1))
        h2, w2 = image2.shape[:2]

    # Create canvas
    h = h1
    w = w1 + gap + w2

    if image1.ndim == 3:
        result = np.ones((h, w, 3), dtype=np.uint8) * 255
        result[:, :w1] = image1
        result[:, w1+gap:] = image2
    else:
        result = np.ones((h, w), dtype=np.uint8) * 255
        result[:, :w1] = image1
        result[:, w1+gap:] = image2

    # Add labels
    if labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, labels[0], (10, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(result, labels[1], (w1 + gap +
                    10, 30), font, 1, (0, 0, 0), 2)

    return result


def create_difference_heatmap(
    image1: np.ndarray,
    image2: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Create difference heatmap between two images.

    Args:
        image1: First image
        image2: Second image
        colormap: OpenCV colormap

    Returns:
        Heatmap visualization
    """
    # Ensure same size
    if image1.shape != image2.shape:
        image2 = resize_image(image2, size=(image1.shape[1], image1.shape[0]))

    # Convert to float
    img1 = image1.astype(np.float32)
    img2 = image2.astype(np.float32)

    # Compute absolute difference
    diff = np.abs(img1 - img2)

    # Convert to grayscale if color
    if diff.ndim == 3:
        diff = np.mean(diff, axis=2)

    # Normalize to 0-255
    diff = (diff / diff.max() * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(diff, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap


def create_edge_overlay(
    image: np.ndarray,
    edge_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
    canny_low: int = 50,
    canny_high: int = 150
) -> np.ndarray:
    """
    Create image with edge overlay.

    Args:
        image: Input image
        edge_color: Color for edges (R, G, B)
        alpha: Overlay transparency
        canny_low: Canny low threshold
        canny_high: Canny high threshold

    Returns:
        Image with edge overlay
    """
    # Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Detect edges
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Create overlay
    overlay = image.copy()
    overlay[edges > 0] = edge_color

    # Blend
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return result


def create_zoom_comparison(
    image1: np.ndarray,
    image2: np.ndarray,
    zoom_rect: Tuple[int, int, int, int],
    zoom_scale: int = 4,
    labels: Tuple[str, str] = ('Before', 'After')
) -> np.ndarray:
    """
    Create zoomed comparison of a specific region.

    Args:
        image1: First image (before)
        image2: Second image (after)
        zoom_rect: (x, y, width, height) region to zoom
        zoom_scale: Zoom magnification
        labels: Image labels

    Returns:
        Comparison with zoomed insets
    """
    x, y, w, h = zoom_rect

    # Extract regions
    region1 = image1[y:y+h, x:x+w]
    region2 = image2[y:y+h, x:x+w]

    # Zoom
    zoom1 = resize_image(region1, scale=zoom_scale, method='nearest')
    zoom2 = resize_image(region2, scale=zoom_scale, method='nearest')

    # Create side by side
    return create_side_by_side(zoom1, zoom2, labels)


# =============================================================================
# Metrics Display
# =============================================================================

def format_metrics(metrics: dict) -> str:
    """
    Format metrics dictionary for display.

    Args:
        metrics: Dictionary of metric values

    Returns:
        Formatted string
    """
    lines = []

    if 'psnr' in metrics:
        lines.append(f"PSNR: {metrics['psnr']:.2f} dB")
    if 'ssim' in metrics:
        lines.append(f"SSIM: {metrics['ssim']:.4f}")
    if 'lpips' in metrics:
        lines.append(f"LPIPS: {metrics['lpips']:.4f}")
    if 'edge_coherence' in metrics:
        lines.append(f"Edge Coherence: {metrics['edge_coherence']:.4f}")

    return "\n".join(lines)


def create_metrics_badge(
    metrics: dict,
    size: Tuple[int, int] = (200, 150)
) -> np.ndarray:
    """
    Create a visual metrics badge.

    Args:
        metrics: Dictionary of metric values
        size: Badge size (width, height)

    Returns:
        Badge image
    """
    w, h = size
    badge = np.ones((h, w, 3), dtype=np.uint8) * 40  # Dark background

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 25
    line_height = 30

    # Title
    cv2.putText(badge, "Metrics", (10, y_offset),
                font, 0.6, (255, 255, 255), 1)
    y_offset += line_height + 5

    # Add metrics
    for key, value in metrics.items():
        if key == 'psnr':
            text = f"PSNR: {value:.2f} dB"
            color = (0, 255, 0) if value > 28 else (
                255, 165, 0) if value > 25 else (255, 0, 0)
        elif key == 'ssim':
            text = f"SSIM: {value:.4f}"
            color = (0, 255, 0) if value > 0.85 else (
                255, 165, 0) if value > 0.8 else (255, 0, 0)
        elif key == 'edge_coherence':
            text = f"Edge Coh: {value:.4f}"
            color = (0, 255, 0) if value > 0.9 else (
                255, 165, 0) if value > 0.85 else (255, 0, 0)
        else:
            text = f"{key}: {value:.4f}"
            color = (255, 255, 255)

        cv2.putText(badge, text, (10, y_offset), font, 0.5, color, 1)
        y_offset += line_height

    return badge


# =============================================================================
# Sample Images
# =============================================================================

def create_sample_satellite_image(size: int = 256) -> np.ndarray:
    """
    Create a synthetic satellite-like image for testing.

    Args:
        size: Image size

    Returns:
        Synthetic satellite image
    """
    image = np.random.rand(size, size, 3) * 0.3 + 0.2

    # Add buildings (rectangular blocks)
    for _ in range(np.random.randint(5, 15)):
        x = np.random.randint(0, size - 30)
        y = np.random.randint(0, size - 30)
        w = np.random.randint(10, 40)
        h = np.random.randint(10, 40)
        color = np.random.rand(3) * 0.3 + 0.4
        image[y:y+h, x:x+w] = color

    # Add roads
    for _ in range(np.random.randint(2, 5)):
        if np.random.rand() > 0.5:
            y = np.random.randint(0, size)
            image[max(0, y-2):min(size, y+2), :] = [0.3, 0.3, 0.35]
        else:
            x = np.random.randint(0, size)
            image[:, max(0, x-2):min(size, x+2)] = [0.3, 0.3, 0.35]

    # Add vegetation
    for _ in range(np.random.randint(2, 6)):
        x = np.random.randint(0, size - 20)
        y = np.random.randint(0, size - 20)
        r = np.random.randint(5, 15)
        yy, xx = np.ogrid[:size, :size]
        mask = ((xx - x - r) ** 2 + (yy - y - r) ** 2) < r ** 2
        image[mask] = [0.2, 0.5 + np.random.rand() * 0.2, 0.15]

    return (image * 255).astype(np.uint8)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing demo utilities...")

    # Create sample image
    sample = create_sample_satellite_image(256)
    print(f"Sample image shape: {sample.shape}")

    # Test resize
    resized = resize_image(sample, scale=0.5)
    print(f"Resized image shape: {resized.shape}")

    # Test difference heatmap
    sample2 = create_sample_satellite_image(256)
    heatmap = create_difference_heatmap(sample, sample2)
    print(f"Heatmap shape: {heatmap.shape}")

    # Test edge overlay
    edges = create_edge_overlay(sample)
    print(f"Edge overlay shape: {edges.shape}")

    # Test metrics badge
    metrics = {'psnr': 28.5, 'ssim': 0.87, 'edge_coherence': 0.92}
    badge = create_metrics_badge(metrics)
    print(f"Metrics badge shape: {badge.shape}")

    print("✓ All utilities tests passed!")
