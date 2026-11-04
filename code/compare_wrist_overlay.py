#!/usr/bin/env python3

import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "physikal_workflow/dex-5"

TAKE = ROOT / (
    "code/stereo/recordings_parallel_v3/"
    "take_20261103_211717_574941"
)

TRIAL = ROOT / (
    "FoundationStereo/"
    "wristband_frame36_depth_take_20261103_211717_574941"
)

ORIGINAL = TRIAL / "wrist_xyz_camera1.csv"

SMOOTHED = (
    TRIAL / "smoothing_test/wrist_xyz_smoothed.csv"
)

CALIB = ROOT / (
    "code/stereo/calibration/"
    "parallel_results_v3/stereo_calibration.npz"
)

OUT = TRIAL / "smoothing_test/overlay_validation"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Calibration
# --------------------------------------------------

with np.load(CALIB) as c:
    K = c["K1"].astype(np.float64)
    D = c["D1"].astype(np.float64)

# Both inputs are already in original Camera 1 coordinates.
# No table/world conversion is needed for this projection.

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


original_rows = read_csv(ORIGINAL)
smoothed_rows = read_csv(SMOOTHED)

N = len(original_rows)

original_xyz = np.full((N, 3), np.nan)
smoothed_xyz = np.full((N, 3), np.nan)

for row in original_rows:
    i = int(row["source_video_frame"])

    if not 0 <= i < N:
        continue

    if row["status"] != "VALID":
        continue

    original_xyz[i] = [
        float(row["X_cam1_m"]),
        float(row["Y_cam1_m"]),
        float(row["Z_cam1_m"]),
    ]

for row in smoothed_rows:
    i = int(row["source_video_frame"])

    if not 0 <= i < N:
        continue

    smoothed_xyz[i] = [
        float(row["smoothed_X_cam1_m"]),
        float(row["smoothed_Y_cam1_m"]),
        float(row["smoothed_Z_cam1_m"]),
    ]


def project(p):
    if not np.isfinite(p).all() or p[2] <= 0:
        return None

    uv, _ = cv2.projectPoints(
        p.reshape(1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        K,
        D
    )

    return uv.reshape(2)


def draw_point(img, uv, color, label):
    if uv is None:
        return

    if not np.isfinite(uv).all():
        return

    h, w = img.shape[:2]
    x, y = np.rint(uv).astype(int)

    if not (0 <= x < w and 0 <= y < h):
        return

    cv2.circle(img, (x, y), 10, (0, 0, 0), 4)
    cv2.circle(img, (x, y), 9, color, 3)

    cv2.putText(
        img,
        label,
        (min(x + 15, w - 160), max(y - 15, 120)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


# --------------------------------------------------
# Projection comparison
# --------------------------------------------------

original_uv = [project(p) for p in original_xyz]
smoothed_uv = [project(p) for p in smoothed_xyz]

pixel_difference = np.full(N, np.nan)
xyz_difference_mm = np.full(N, np.nan)

for i in range(N):
    if original_uv[i] is None or smoothed_uv[i] is None:
        continue

    pixel_difference[i] = np.linalg.norm(
        original_uv[i] - smoothed_uv[i]
    )

    xyz_difference_mm[i] = (
        np.linalg.norm(
            original_xyz[i] - smoothed_xyz[i]
        ) * 1000
    )


# --------------------------------------------------
# Video source
# --------------------------------------------------

cap = cv2.VideoCapture(str(TAKE / "camera1.avi"))

if not cap.isOpened():
    raise RuntimeError("Cannot open camera1.avi")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 10.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if video_frames < N:
    raise RuntimeError(
        f"Video contains {video_frames} frames "
        f"but CSV contains {N}"
    )

print("Video size:", width, "x", height)
print("Video FPS:", fps)
print("Frames:", N)


def make_writer(filename):
    writer = cv2.VideoWriter(
        str(OUT / filename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create video: {filename}"
        )

    return writer


full_writer = make_writer("full_wrist_comparison.mp4")
focus_writer = make_writer("frames_36_to_85.mp4")

# --------------------------------------------------
# Generate synchronized overlays
# --------------------------------------------------

frame = 0

while frame < N:
    ok, img = cap.read()

    if not ok:
        break

    original = original_uv[frame]
    smooth = smoothed_uv[frame]

    # Yellow: original FoundationStereo
    draw_point(
        img,
        original,
        (0, 255, 255),
        "Original FS"
    )

    # Green: smoothed wrist
    draw_point(
        img,
        smooth,
        (0, 255, 0),
        "Smoothed"
    )

    # Red: projected difference
    if original is not None and smooth is not None:
        p1 = tuple(np.rint(original).astype(int))
        p2 = tuple(np.rint(smooth).astype(int))

        cv2.line(
            img,
            p1,
            p2,
            (0, 0, 255),
            2
        )


    # Small frame counter at the bottom-left.
    # No black panel or text over the wristband.
    cv2.putText(
        img,
        f"Frame {frame:03d}",
        (15, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    full_writer.write(img)

    if 36 <= frame <= 85:
        focus_writer.write(img)

    frame += 1

cap.release()
full_writer.release()
focus_writer.release()


# --------------------------------------------------
# Pixel displacement graph
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(15, 5))

frames = np.arange(N)

ax.plot(
    frames,
    pixel_difference,
    label="Original vs smoothed reprojection"
)

ax.axvspan(
    36, 62,
    alpha=0.15,
    label="Estimated-depth interval"
)

ax.axvline(
    63,
    linestyle="--",
    label="Measured-depth section begins"
)

ax.set_xlabel("Source video frame")
ax.set_ylabel("Projected displacement (pixels)")
ax.set_title("Change in image position after smoothing")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "pixel_displacement.png",
    dpi=160
)
plt.close(fig)


# --------------------------------------------------
# Export diagnostics
# --------------------------------------------------

with (OUT / "projection_comparison.csv").open(
    "w", newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "frame",
        "original_depth_m",
        "smoothed_depth_m",
        "original_u",
        "original_v",
        "smoothed_u",
        "smoothed_v",
        "pixel_displacement",
        "xyz_correction_mm"
    ])

    for i in range(N):
        o = original_uv[i]
        s = smoothed_uv[i]

        writer.writerow([
            i,
            original_xyz[i, 2],
            smoothed_xyz[i, 2],
            *(o if o is not None else [np.nan, np.nan]),
            *(s if s is not None else [np.nan, np.nan]),
            pixel_difference[i],
            xyz_difference_mm[i]
        ])


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n========== REPROJECTION DIAGNOSTIC ==========")

for start, end in [
    (36, 62),
    (63, 85),
    (86, 282),
    (36, 282)
]:

    values = pixel_difference[start:end+1]
    values = values[np.isfinite(values)]

    print(f"\nFrames {start}-{end}")

    if not len(values):
        print("No comparable points")
        continue

    print("Compared frames:", len(values))
    print("Median displacement:", round(float(np.median(values)), 2), "px")
    print("P90 displacement:", round(float(np.percentile(values, 90)), 2), "px")
    print("Maximum displacement:", round(float(np.max(values)), 2), "px")

print("\n========== OUTPUTS ==========")
print("Full video:", OUT / "full_wrist_comparison.mp4")
print("Focused video:", OUT / "frames_36_to_85.mp4")
print("Graph:", OUT / "pixel_displacement.png")
print("CSV:", OUT / "projection_comparison.csv")

print("\nOriginal reconstruction unchanged.")
