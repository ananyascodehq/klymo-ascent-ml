"""
KLYMO ASCENT ML - Geospatial Super-Resolution Pipeline
======================================================

A multi-stage deep learning pipeline combining Generative Adversarial Networks 
with physics-informed constraints for 8x satellite image super-resolution 
(10m → 1.25m/pixel) with hallucination prevention.

Key Components:
- ESRGAN Generator with Geospatial Attention Module (GAM)
- Multi-Scale Discriminator for structural validation
- Spatial Consistency Loss for hallucination prevention
"""

__version__ = "1.0.0"
__author__ = "KLYMO Team"
