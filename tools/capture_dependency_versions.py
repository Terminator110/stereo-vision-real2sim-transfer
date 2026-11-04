#!/usr/bin/env python3
"""Capture installed package versions as an audit, not a portable lockfile."""
import importlib.metadata as metadata
import sys
from pathlib import Path

names = [
    'PyYAML','numpy','scipy','opencv-python','mediapipe','mujoco',
    'pandas','Pillow','matplotlib','tqdm','torch','torchvision',
    'transformers','huggingface-hub','safetensors','einops',
    'hydra-core','omegaconf','scikit-image','scikit-learn','imageio',
    'imageio-ffmpeg','trimesh',
]
print(f"Interpreter: {sys.executable}")
print(f"Python: {sys.version.split()[0]}")
missing = []
for name in names:
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        missing.append(name)
print("Missing (may be optional in your build):", ", ".join(missing) or "none")
