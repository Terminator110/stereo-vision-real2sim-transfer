import pandas as pd
import numpy as np
from pathlib import Path

path = Path(__file__).resolve().parent / "wrist_trajectory.csv"
df = pd.read_csv(path)
valid = df[df.valid == 1]

print("Total frames:", len(df))
print("Valid frames:", len(valid))
print("Valid percentage:", round(len(valid) / len(df) * 100, 2) if len(df) else 0)

if len(valid) > 1:
    xyz = valid[["x_cam1_m", "y_cam1_m", "z_cam1_m"]] * 1000
    print("Mean XYZ (mm):", xyz.mean().round(2).to_list())
    print("STD XYZ (mm):", xyz.std().round(2).to_list())
    print("Range XYZ (mm):", (xyz.max() - xyz.min()).round(2).to_list())
    print("Mean reprojection error (px):",
          round(valid.mean_reprojection_px.mean(), 3))
    print("Mean reconstructed tag edge (mm):",
          round(valid.mean_tag_edge_mm.mean(), 2))
