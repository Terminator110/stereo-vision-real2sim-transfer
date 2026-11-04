#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter, gaussian_filter1d

ROOT = Path.home() / "physikal_workflow/dex-5"

TRIAL = ROOT / (
    "FoundationStereo/"
    "wristband_frame36_depth_take_20261103_211717_574941"
)

OUT = TRIAL / "smoothing_test"
OUT.mkdir(parents=True, exist_ok=True)

INPUT = TRIAL / "wrist_xyz_camera1.csv"

START = 36
ANCHOR_END = 62
STABLE_START = 65
STABLE_END = 85
SIGMA = 2.0
MAX_RESIDUAL_M = 0.070

with INPUT.open(newline="") as f:
    rows = list(csv.DictReader(f))

N = len(rows)
assert N == 283, f"Expected 283 frames, got {N}"

cam = np.full((N, 3), np.nan)
status = ["INVALID"] * N

for r in rows:
    i = int(r["source_video_frame"])
    status[i] = r["status"]

    if r["status"] != "VALID":
        continue

    cam[i] = [
        float(r["X_cam1_m"]),
        float(r["Y_cam1_m"]),
        float(r["Z_cam1_m"])
    ]

valid = np.isfinite(cam).all(axis=1)

# --------------------------------------------------
# 1. Find stable reference depth
# --------------------------------------------------

reference = np.arange(STABLE_START, STABLE_END + 1)

good = reference[
    valid[reference]
    & (cam[reference, 2] > 0.5)
    & (cam[reference, 2] < 1.2)
]

if len(good) < 10:
    raise RuntimeError(
        "Insufficient stable reference measurements. "
        "Cannot estimate early depth safely."
    )

anchor_z = float(np.median(cam[good, 2]))

print("========== DEPTH ANCHOR ==========")
print("Reference frames:", good.tolist())
print("Median reference Camera 1 Z:", round(anchor_z, 4), "m")

# --------------------------------------------------
# 2. Correct early depth using the original camera ray
# --------------------------------------------------

corrected = cam.copy()
sources = ["MISSING"] * N

for i in range(START, N):
    if not valid[i]:
        continue

    if i <= ANCHOR_END:
        # Keep image-space direction but replace unreliable depth.
        ray = cam[i] / cam[i, 2]
        corrected[i] = ray * anchor_z
        sources[i] = "ESTIMATED_ANCHORED_DEPTH"
    else:
        sources[i] = "FOUNDATIONSTEREO"

# Require every frame in the requested section to have a
# valid camera ray. Never silently fabricate missing points.
missing = [
    i for i in range(START, N)
    if not np.isfinite(corrected[i]).all()
]

if missing:
    raise RuntimeError(
        f"Missing frames remain: {missing}. "
        "This test requires continuous tracked measurements."
    )

# --------------------------------------------------
# 3. Robust smoothing
# --------------------------------------------------

section = corrected[START:].copy()

# Median-based local reference rejects isolated excursions.
local_median = median_filter(
    section,
    size=(7, 1),
    mode="nearest"
)

residual = section - local_median

residual_norm = np.linalg.norm(
    residual, axis=1
)

# Soft-limit large deviations instead of discarding frames.
scale = np.minimum(
    1.0,
    MAX_RESIDUAL_M / np.maximum(residual_norm, 1e-9)
)

robust = local_median + residual * scale[:, None]

# Smooth all three axes, including the early/late transition.
smoothed_section = gaussian_filter1d(
    robust,
    sigma=SIGMA,
    axis=0,
    mode="nearest"
)

smoothed = np.full_like(cam, np.nan)
smoothed[START:] = smoothed_section

# --------------------------------------------------
# 4. Convert to calibrated table and MuJoCo world XYZ
# --------------------------------------------------

TABLE = ROOT / (
    "code/stereo/calibration/"
    "parallel_results_v3/table_frame_multi.npz"
)

with np.load(TABLE) as z:
    Rct = z["R_cam1_to_table"].astype(float)
    tct = z["t_cam1_to_table"].reshape(3).astype(float)

yaw = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1]
], dtype=float)

origin = np.array([0.385, -0.120, 0.832])

table_xyz = np.full_like(smoothed, np.nan)
world_xyz = np.full_like(smoothed, np.nan)

for i in range(START, N):
    table_xyz[i] = Rct @ smoothed[i] + tct
    world_xyz[i] = yaw @ table_xyz[i] + origin

# --------------------------------------------------
# 5. Save CSV
# --------------------------------------------------

output_csv = OUT / "wrist_xyz_smoothed.csv"

with output_csv.open("w", newline="") as f:
    w = csv.writer(f)

    w.writerow([
        "source_video_frame",
        "source",
        "original_X_cam1_m",
        "original_Y_cam1_m",
        "original_Z_cam1_m",
        "smoothed_X_cam1_m",
        "smoothed_Y_cam1_m",
        "smoothed_Z_cam1_m",
        "X_table_m",
        "Y_table_m",
        "Z_table_m",
        "X_world_m",
        "Y_world_m",
        "Z_world_m"
    ])

    for i in range(N):
        w.writerow([
            i,
            sources[i],
            *cam[i],
            *smoothed[i],
            *table_xyz[i],
            *world_xyz[i]
        ])

# --------------------------------------------------
# 6. Diagnostic charts
# --------------------------------------------------

fig, axes = plt.subplots(
    3, 1,
    figsize=(15, 11),
    sharex=True
)

frames = np.arange(N)

for j, axis in enumerate("XYZ"):
    axes[j].plot(
        frames,
        cam[:, j],
        label="Original FoundationStereo",
        alpha=0.65
    )

    axes[j].plot(
        frames,
        smoothed[:, j],
        label="Depth-stabilized trajectory",
        linewidth=2
    )

    axes[j].axvspan(
        START, ANCHOR_END,
        alpha=0.12,
        label="Estimated-depth section" if j == 0 else None
    )

    axes[j].set_ylabel(f"Camera {axis} (m)")
    axes[j].grid(True, alpha=0.3)
    axes[j].legend()

axes[-1].set_xlabel("Source video frame")

fig.suptitle("Original vs depth-stabilized wrist trajectory")
fig.tight_layout()

fig.savefig(
    OUT / "xyz_comparison.png",
    dpi=160
)

plt.close(fig)

# --------------------------------------------------
# 7. Displacement comparison
# --------------------------------------------------

original_step = np.linalg.norm(
    np.diff(cam[START:], axis=0),
    axis=1
) * 1000

smoothed_step = np.linalg.norm(
    np.diff(smoothed[START:], axis=0),
    axis=1
) * 1000

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(
    frames[START + 1:],
    original_step,
    label="Original displacement"
)

ax.plot(
    frames[START + 1:],
    smoothed_step,
    label="Smoothed displacement"
)

ax.set_xlabel("Source video frame")
ax.set_ylabel("Displacement (mm/frame)")
ax.set_title("Frame-to-frame wrist movement")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    OUT / "movement_comparison.png",
    dpi=160
)

plt.close(fig)

# --------------------------------------------------
# 8. Summary
# --------------------------------------------------

print("\n========== SMOOTHING DIAGNOSTIC ==========")

print("Total frames:", N)
print("Tracked section:", START, "to", N - 1)
print("Original valid tracked frames:", int(valid[START:].sum()))
print("Smoothed trajectory frames:", int(
    np.isfinite(smoothed).all(axis=1).sum()
))

print("\nOriginal displacement:")
print("Median:", round(float(np.median(original_step)), 2), "mm")
print("P90:", round(float(np.percentile(original_step, 90)), 2), "mm")
print("Maximum:", round(float(np.max(original_step)), 2), "mm")

print("\nSmoothed displacement:")
print("Median:", round(float(np.median(smoothed_step)), 2), "mm")
print("P90:", round(float(np.percentile(smoothed_step, 90)), 2), "mm")
print("Maximum:", round(float(np.max(smoothed_step)), 2), "mm")

print("\nEstimated-depth frames:", ANCHOR_END - START + 1)
print("Output:", output_csv)
print("Charts:", OUT)
print("\nOriginal FoundationStereo data unchanged.")
