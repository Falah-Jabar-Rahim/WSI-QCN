#!/usr/bin/env python3
"""
Simple installation test for the H&E Color Normalization pipeline.
"""

import sys

print("=" * 60)
print("Testing installation...")
print("=" * 60)

# ---------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------
print(f"Python: {sys.version.split()[0]}")

# ---------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------
try:
    import numpy as np

    print(f"✓ NumPy: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")

# ---------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------
try:
    import cv2

    print(f"✓ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV: {e}")

# ---------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------
try:
    import torch

    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch: {e}")

# ---------------------------------------------------------------------
# Torchvision
# ---------------------------------------------------------------------
try:
    import torchvision

    print(f"✓ Torchvision: {torchvision.__version__}")
except Exception as e:
    print(f"✗ Torchvision: {e}")

# ---------------------------------------------------------------------
# TorchStain
# ---------------------------------------------------------------------
try:
    import torchstain

    print(f"✓ TorchStain: {torchstain.__version__}")
except Exception as e:
    print(f"✗ TorchStain: {e}")

# ---------------------------------------------------------------------
# TIA Toolbox
# ---------------------------------------------------------------------
try:
    import tiatoolbox

    print(f"✓ TIAToolbox: {tiatoolbox.__version__}")
except Exception as e:
    print(f"✗ TIAToolbox: {e}")

# ---------------------------------------------------------------------
# StainTools
# ---------------------------------------------------------------------
try:
    import staintools

    print("✓ StainTools")
except Exception as e:
    print(f"✗ StainTools: {e}")

# ---------------------------------------------------------------------
# Kornia
# ---------------------------------------------------------------------
try:
    import kornia

    print(f"✓ Kornia: {kornia.__version__}")
except Exception as e:
    print(f"✗ Kornia: {e}")

print("=" * 60)
print("Installation test completed.")
print("=" * 60)
