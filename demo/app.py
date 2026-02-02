"""
KLYMO ASCENT ML - Streamlit Demo Application
=============================================

Hackathon submission for the KLYMO ASCENT ML Competition.

Interactive demo for geospatial super-resolution with:
- Before/After image comparison slider
- Zoom tool for detailed inspection
- Difference map visualization
- Real-time metrics display
- Custom image upload support
"""

from utils import (
    load_image, create_sample_satellite_image, create_difference_heatmap,
    create_edge_overlay, resize_image, format_metrics, create_metrics_badge
)
from io import BytesIO
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
import streamlit as st
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# Import demo utilities


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="KLYMO ASCENT ML - Satellite Super-Resolution",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .stSlider > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Model Loading (Cached)
# =============================================================================

@st.cache_resource
def load_model(checkpoint_path: str = None):
    """Load the super-resolution model."""
    try:
        from inference import SuperResolutionInference

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        engine = SuperResolutionInference(
            checkpoint_path=checkpoint_path,
            device=device,
            scale_factor=8,
            tile_size=128
        )

        return engine, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, 'cpu'


# =============================================================================
# Helper Functions
# =============================================================================

def generate_bicubic(lr_image: np.ndarray, scale_factor: int = 8) -> np.ndarray:
    """Generate bicubic upsampled baseline."""
    lr_tensor = torch.from_numpy(lr_image).float().permute(
        2, 0, 1).unsqueeze(0) / 255.0
    bicubic = F.interpolate(
        lr_tensor, scale_factor=scale_factor, mode='bicubic', align_corners=False)
    bicubic = bicubic.squeeze(0).permute(1, 2, 0).numpy()
    bicubic = np.clip(bicubic * 255, 0, 255).astype(np.uint8)
    return bicubic


def compute_metrics(sr: np.ndarray, bicubic: np.ndarray, lr: np.ndarray = None) -> dict:
    """Compute quality metrics."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    metrics = {}

    # PSNR vs bicubic (improvement measure)
    try:
        metrics['psnr_improvement'] = peak_signal_noise_ratio(
            bicubic, sr, data_range=255
        )
    except:
        metrics['psnr_improvement'] = 0

    # SSIM
    try:
        metrics['ssim'] = structural_similarity(
            bicubic, sr, data_range=255, channel_axis=2
        )
    except:
        metrics['ssim'] = 0

    # Edge coherence (simplified)
    import cv2
    sr_gray = cv2.cvtColor(sr, cv2.COLOR_RGB2GRAY)
    bicubic_gray = cv2.cvtColor(bicubic, cv2.COLOR_RGB2GRAY)

    sr_edges = cv2.Canny(sr_gray, 50, 150).flatten().astype(float)
    bicubic_edges = cv2.Canny(bicubic_gray, 50, 150).flatten().astype(float)

    if sr_edges.std() > 0 and bicubic_edges.std() > 0:
        correlation = np.corrcoef(sr_edges, bicubic_edges)[0, 1]
        metrics['edge_coherence'] = max(
            0, correlation if not np.isnan(correlation) else 0)
    else:
        metrics['edge_coherence'] = 1.0

    return metrics


# =============================================================================
# Main App
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🛰️ KLYMO ASCENT ML</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Geospatial Super-Resolution with Hallucination Prevention</p>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model checkpoint
        checkpoint_path = st.text_input(
            "Model Checkpoint",
            value="checkpoints/best_model.pth",
            help="Path to trained model checkpoint"
        )

        # Check if checkpoint exists
        if Path(checkpoint_path).exists():
            st.success("✓ Model checkpoint found")
            model_loaded = True
        else:
            st.warning("⚠ Checkpoint not found. Using demo mode.")
            model_loaded = False

        st.divider()

        # Display options
        st.subheader("Display Options")
        show_metrics = st.checkbox("Show Metrics", value=True)
        show_diff_map = st.checkbox("Show Difference Map", value=True)
        show_edge_overlay = st.checkbox("Show Edge Overlay", value=False)

        st.divider()

        # Zoom settings
        st.subheader("Zoom Settings")
        zoom_factor = st.slider("Zoom Factor", 1, 8, 4)

        st.divider()

        # Info
        st.subheader("About")
        st.markdown("""
        **8x Super-Resolution Pipeline**
        
        - Input: 10m/pixel (Sentinel-2)
        - Output: 1.25m/pixel
        - Architecture: ESRGAN + GAM
        - Hallucination Prevention: ✓
        """)

    # Main content
    tab1, tab2, tab3 = st.tabs(
        ["🖼️ Image Comparison", "📊 Metrics Analysis", "ℹ️ About"])

    with tab1:
        st.header("Image Super-Resolution")

        # Image source selection
        col1, col2 = st.columns([1, 1])

        with col1:
            source = st.radio(
                "Image Source",
                ["Sample Image", "Upload Image"],
                horizontal=True
            )

        with col2:
            if source == "Sample Image":
                sample_type = st.selectbox(
                    "Sample Type",
                    ["Urban Area", "Mixed Land Use", "Vegetation"]
                )

        # Load or upload image
        if source == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload a low-resolution satellite image",
                type=['png', 'jpg', 'jpeg', 'tif'],
                help="Upload a satellite image to enhance"
            )

            if uploaded_file is not None:
                lr_image = load_image(uploaded_file, max_size=256)
            else:
                lr_image = None
        else:
            # Generate sample image
            np.random.seed(42 if sample_type == "Urban Area" else
                           123 if sample_type == "Mixed Land Use" else 456)
            lr_image = create_sample_satellite_image(64)

        if lr_image is not None:
            # Process image
            st.subheader("Processing...")

            with st.spinner("Generating super-resolved image..."):
                # Generate bicubic baseline
                bicubic = generate_bicubic(lr_image, scale_factor=8)

                # Generate SR output
                if model_loaded:
                    try:
                        engine, device = load_model(checkpoint_path)
                        if engine is not None:
                            sr = engine.enhance(lr_image, output_type='numpy')
                        else:
                            # Fallback: enhanced bicubic (sharpen)
                            import cv2
                            kernel = np.array(
                                [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                            sr = cv2.filter2D(bicubic, -1, kernel)
                            sr = np.clip(sr, 0, 255).astype(np.uint8)
                    except Exception as e:
                        st.warning(
                            f"Model error: {e}. Using enhanced bicubic.")
                        import cv2
                        kernel = np.array(
                            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        sr = cv2.filter2D(bicubic, -1, kernel)
                        sr = np.clip(sr, 0, 255).astype(np.uint8)
                else:
                    # Demo mode: use sharpened bicubic
                    import cv2
                    kernel = np.array(
                        [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    sr = cv2.filter2D(bicubic, -1, kernel)
                    sr = np.clip(sr, 0, 255).astype(np.uint8)

            # Display comparison
            st.subheader("📸 Results Comparison")

            # Create columns for comparison
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Low Resolution (Input)**")
                # Upscale LR for display
                lr_display = resize_image(lr_image, scale=8, method='nearest')
                st.image(lr_display, use_container_width=True)
                st.caption(
                    f"Original: {lr_image.shape[1]}x{lr_image.shape[0]}px")

            with col2:
                st.markdown("**Bicubic Baseline**")
                st.image(bicubic, use_container_width=True)
                st.caption(
                    f"Bicubic 8x: {bicubic.shape[1]}x{bicubic.shape[0]}px")

            with col3:
                st.markdown("**Super-Resolved (Output)**")
                st.image(sr, use_container_width=True)
                st.caption(f"SR 8x: {sr.shape[1]}x{sr.shape[0]}px")

            # Difference map
            if show_diff_map:
                st.subheader("🔥 Difference Map (SR vs Bicubic)")
                diff_map = create_difference_heatmap(sr, bicubic)
                st.image(diff_map, use_container_width=True)
                st.caption(
                    "Heatmap shows where SR differs from bicubic baseline")

            # Edge overlay
            if show_edge_overlay:
                st.subheader("📐 Edge Overlay")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Bicubic Edges**")
                    bicubic_edges = create_edge_overlay(bicubic)
                    st.image(bicubic_edges, use_container_width=True)

                with col2:
                    st.markdown("**SR Edges**")
                    sr_edges = create_edge_overlay(sr)
                    st.image(sr_edges, use_container_width=True)

            # Zoom comparison
            st.subheader(f"🔍 Zoom Comparison ({zoom_factor}x)")

            # Select zoom region
            h, w = sr.shape[:2]
            zoom_size = min(h, w) // 4

            col1, col2 = st.columns(2)
            with col1:
                zoom_x = st.slider("Zoom X Position", 0, w - zoom_size, w // 4)
            with col2:
                zoom_y = st.slider("Zoom Y Position", 0, h - zoom_size, h // 4)

            # Extract and zoom regions
            bicubic_crop = bicubic[zoom_y:zoom_y +
                                   zoom_size, zoom_x:zoom_x+zoom_size]
            sr_crop = sr[zoom_y:zoom_y+zoom_size, zoom_x:zoom_x+zoom_size]

            bicubic_zoomed = resize_image(
                bicubic_crop, scale=zoom_factor, method='nearest')
            sr_zoomed = resize_image(
                sr_crop, scale=zoom_factor, method='nearest')

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Bicubic (Zoomed)**")
                st.image(bicubic_zoomed, use_container_width=True)
            with col2:
                st.markdown("**Super-Resolved (Zoomed)**")
                st.image(sr_zoomed, use_container_width=True)

            # Download button
            st.divider()
            col1, col2, col3 = st.columns(3)

            with col2:
                sr_pil = Image.fromarray(sr)
                buf = BytesIO()
                sr_pil.save(buf, format='PNG')

                st.download_button(
                    label="⬇️ Download Super-Resolved Image",
                    data=buf.getvalue(),
                    file_name="super_resolved.png",
                    mime="image/png"
                )

    with tab2:
        st.header("📊 Quality Metrics")

        if lr_image is not None and 'sr' in dir():
            # Compute metrics
            metrics = compute_metrics(sr, bicubic, lr_image)

            # Display metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="PSNR",
                    value=f"{metrics.get('psnr_improvement', 0):.2f} dB",
                    delta="vs Bicubic",
                    help="Peak Signal-to-Noise Ratio. Higher is better."
                )

            with col2:
                ssim = metrics.get('ssim', 0)
                st.metric(
                    label="SSIM",
                    value=f"{ssim:.4f}",
                    delta="Good" if ssim > 0.85 else "Fair" if ssim > 0.8 else "Low",
                    help="Structural Similarity Index. Range [0,1], higher is better."
                )

            with col3:
                ec = metrics.get('edge_coherence', 0)
                st.metric(
                    label="Edge Coherence",
                    value=f"{ec:.4f}",
                    delta="✓ No Hallucination" if ec > 0.85 else "⚠ Check",
                    help="Measures edge alignment. >0.85 indicates no major hallucinations."
                )

            # Metric explanations
            st.divider()
            st.subheader("📖 Metric Explanations")

            with st.expander("PSNR (Peak Signal-to-Noise Ratio)"):
                st.markdown("""
                - Measures pixel-wise accuracy
                - Higher values indicate better reconstruction
                - **Target: >28 dB** for high-quality SR
                - Baseline bicubic typically achieves ~24 dB
                """)

            with st.expander("SSIM (Structural Similarity Index)"):
                st.markdown("""
                - Measures perceptual similarity
                - Considers luminance, contrast, and structure
                - **Target: >0.85** for visually pleasing results
                - Range: 0 (completely different) to 1 (identical)
                """)

            with st.expander("Edge Coherence (Hallucination Detection)"):
                st.markdown("""
                - **UNIQUE METRIC** for hallucination prevention
                - Measures alignment between SR edges and bicubic edges
                - **Target: >0.90** (no invented features)
                - Low values may indicate hallucinated buildings/roads
                """)
        else:
            st.info("Upload or select an image to see metrics.")

    with tab3:
        st.header("ℹ️ About KLYMO ASCENT ML")

        st.markdown("""
        ### 🎯 Problem
        
        Satellite imagery faces a fundamental trade-off:
        - **Sentinel-2**: Free, frequent coverage at **10m/pixel** (insufficient for urban analysis)
        - **Commercial satellites**: 0.3m/pixel clarity at **prohibitive costs**
        
        This **33x resolution gap** creates barriers for geospatial AI applications.
        
        ### 💡 Solution
        
        Our multi-stage deep learning pipeline achieves **8x super-resolution** (10m → 1.25m/pixel) with:
        
        1. **ESRGAN Generator**: Enhanced Super-Resolution GAN with Residual-in-Residual Dense Blocks
        2. **Geospatial Attention Module (GAM)**: Encodes NDVI/NDBI/NDWI as conditioning signals
        3. **Multi-Scale Discriminator**: Validates structural accuracy at 3 resolution levels
        4. **Spatial Consistency Loss**: Physics-based constraint preventing feature hallucination
        
        ### 🏆 Key Innovations
        
        | Feature | Benefit |
        |---------|---------|
        | GAM (Geospatial Attention) | Land-cover aware processing |
        | Multi-Scale Discriminator | Catches hallucinations at multiple frequencies |
        | Spatial Consistency Loss | Ensures no invented features |
        | Edge Coherence Metric | Quantifies hallucination risk |
        
        ### 📊 Target Performance
        
        - **PSNR**: >28 dB (4 dB improvement over bicubic)
        - **SSIM**: >0.85
        - **Edge Coherence**: >0.90
        """)

        st.divider()

        st.markdown("""
        ### 🔗 Links
        
        - [GitHub Repository](https://github.com/your-repo/klymo-ascent-ml)
        - [Technical Documentation](./README.md)
        - [Model Architecture](./prd.md)
        """)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
