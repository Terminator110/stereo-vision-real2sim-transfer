#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "physikal_workflow/dex-5"

TAKE_ID = "take_20261103_211944_200460"

TRIAL = ROOT / "FoundationStereo" / (
    "wristband_batch_" + TAKE_ID
)

DEMO = ROOT / "outputs/reconstructed_demos" / TAKE_ID
RESTORED = ROOT / "outputs/fs157_comparison"

TAKE = ROOT / "code/stereo/recordings_parallel_v3" / TAKE_ID

CALIB = ROOT / (
    "code/stereo/calibration/parallel_results_v3/"
    "stereo_calibration.npz"
)

TABLE = ROOT / (
    "code/stereo/calibration/parallel_results_v3/"
    "table_frame_multi.npz"
)

OUT = ROOT / "outputs/wrist_mapping_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

ORIGIN = np.array([0.385, -0.120, 0.832])
YAW = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1],
], dtype=float)

N = 186
FPS = 20.0


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def xyz(row, keys):
    try:
        v = np.array([float(row[k]) for k in keys])
        return v if np.isfinite(v).all() else None
    except (ValueError, KeyError, TypeError):
        return None


with np.load(TABLE) as d:
    Rct = d["R_cam1_to_table"].astype(float)
    tct = d["t_cam1_to_table"].reshape(3).astype(float)

with np.load(CALIB) as d:
    K = d["K1"].astype(float)
    D = d["D1"].astype(float)
    R1 = d["R1"].astype(float)
    P1 = d["P1"].astype(float)
    image_size = tuple(map(int, d["image_size"]))


# -------------------------------------------------
# Load independent XYZ trajectories
# -------------------------------------------------

fs = np.full((N, 3), np.nan)
agent = np.full((N, 3), np.nan)
restored = np.full((N, 3), np.nan)

fs_cam = np.full((N, 3), np.nan)

rows_fs = read_csv(TRIAL / "wrist_xyz_camera1.csv")

for row in rows_fs:
    i = int(row["source_video_frame"])
    if not 0 <= i < N or row["status"] != "VALID":
        continue

    p = xyz(
        row,
        ("X_cam1_m", "Y_cam1_m", "Z_cam1_m")
    )

    if p is None:
        continue

    fs_cam[i] = p
    fs[i] = Rct @ p + tct


def load_palm(path, result):
    rows = read_csv(path)

    for row in rows:
        i = int(row["source_video_frame"])

        if not 0 <= i < N:
            continue

        p = xyz(
            row,
            ("X_table_m", "Y_table_m", "Z_table_m")
        )

        if p is not None:
            result[i] = p

    return rows


rows_agent = load_palm(
    DEMO / "trial/palm_pose_6dof.csv", agent
)

rows_restored = load_palm(
    RESTORED / "palm_pose_6dof.csv", restored
)


def valid(a):
    return np.isfinite(a).all(axis=1)


def world(a):
    return (YAW @ a.T).T + ORIGIN


fs_world = world(fs)
agent_world = world(agent)
restored_world = world(restored)


# -------------------------------------------------
# Camera projection
# -------------------------------------------------

def project_table(p):
    if not np.isfinite(p).all():
        return None

    camera_p = Rct.T @ (p - tct)

    if camera_p[2] <= 0:
        return None

    uv, _ = cv2.projectPoints(
        camera_p.reshape(1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        K,
        D,
    )

    return uv.reshape(2)


# Calibrated inverse mapping:
# prepared left image is scaled and rotated 90 CCW.
mapx, mapy = cv2.initUndistortRectifyMap(
    K, D, R1, P1[:, :3],
    image_size, cv2.CV_32FC1
)

raw_w, raw_h = image_size

def observed_mask_raw(i):
    mask_path = TRIAL / "masks" / f"{i:05d}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None

    yy, xx = np.nonzero(mask > 127)

    if len(xx) < 10:
        return None

    # Coordinates in the prepared rotated image.
    u = float(np.median(xx))
    v = float(np.median(yy))

    # Reverse resize and 90-degree CCW rotation.
    scale_x = mask.shape[0] / raw_w
    scale_y = mask.shape[1] / raw_h

    rect_x = raw_w - 1 - v / scale_x
    rect_y = u / scale_y

    xi = int(round(rect_x))
    yi = int(round(rect_y))

    if not (0 <= xi < raw_w and 0 <= yi < raw_h):
        return None

    return np.array([mapx[yi, xi], mapy[yi, xi]])


# -------------------------------------------------
# Per-frame diagnostics
# -------------------------------------------------

def distance(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    return float(np.linalg.norm(a-b) * 1000)


def projected_error(p, observed):
    uv = project_table(p)

    if uv is None or observed is None:
        return np.nan

    return float(np.linalg.norm(uv-observed))


diagnostics = []

for i in range(N):
    observed = observed_mask_raw(i)

    row = {
        "frame": i,
        "time_s": i / FPS,
        "fs_valid": int(valid(fs)[i]),
        "agent_valid": int(valid(agent)[i]),
        "restored_valid": int(valid(restored)[i]),
        "fs_agent_difference_mm": distance(fs[i], agent[i]),
        "fs_restored_difference_mm": distance(fs[i], restored[i]),
        "fs_projection_mask_error_px":
            projected_error(fs[i], observed),
        "agent_projection_mask_error_px":
            projected_error(agent[i], observed),
        "restored_projection_mask_error_px":
            projected_error(restored[i], observed),
    }

    for name, arr in (
        ("fs", fs),
        ("agent", agent),
        ("restored", restored),
    ):
        for j, axis in enumerate("xyz"):
            row[f"{name}_{axis}_table_m"] = arr[i, j]

        if i:
            row[f"{name}_step_mm"] = distance(
                arr[i], arr[i-1]
            )
        else:
            row[f"{name}_step_mm"] = np.nan

        w = world(arr[i:i+1])[0]

        for j, axis in enumerate("xyz"):
            row[f"{name}_{axis}_world_m"] = w[j]

    diagnostics.append(row)


csv_path = OUT / "trajectory_mapping.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(diagnostics[0])
    )
    writer.writeheader()
    writer.writerows(diagnostics)


# -------------------------------------------------
# XYZ plots
# -------------------------------------------------

t = np.arange(N) / FPS

fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

for j, axis in enumerate("XYZ"):
    ax = axes[j]

    ax.plot(t, fs[:, j]*1000, label="FoundationStereo")
    ax.plot(t, agent[:, j]*1000, label="Agent")
    ax.plot(
        t, restored[:, j]*1000,
        label="Restored FS157", linestyle="--"
    )

    ax.set_ylabel(f"Table {axis} (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Time (s)")

fig.suptitle("Wrist XYZ — Table-coordinate comparison")
fig.tight_layout()
fig.savefig(OUT / "xyz_table_comparison.png", dpi=160)
plt.close(fig)


fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

for j, axis in enumerate("XYZ"):
    ax = axes[j]

    ax.plot(
        t, fs_world[:, j]*1000,
        label="FoundationStereo world"
    )
    ax.plot(
        t, agent_world[:, j]*1000,
        label="Agent world"
    )
    ax.plot(
        t, restored_world[:, j]*1000,
        label="Restored world", linestyle="--"
    )

    ax.set_ylabel(f"World {axis} (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Time (s)")

fig.suptitle("Wrist XYZ — MuJoCo world-coordinate mapping")
fig.tight_layout()
fig.savefig(OUT / "xyz_world_comparison.png", dpi=160)
plt.close(fig)


# Step-size diagnostic
fig, ax = plt.subplots(figsize=(15, 5))

for name, array in (
    ("FoundationStereo", fs),
    ("Agent", agent),
    ("Restored", restored),
):
    steps = np.full(N, np.nan)

    for i in range(1, N):
        steps[i] = distance(array[i], array[i-1])

    ax.plot(t, steps, label=name)

ax.axhline(
    25, linestyle="--",
    label="25 mm/frame inspection threshold"
)

ax.set_title("Frame-to-frame wrist displacement")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Step (mm)")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(OUT / "xyz_frame_jumps.png", dpi=160)
plt.close(fig)


# 3D trajectory
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")

for name, arr in (
    ("FoundationStereo", fs_world),
    ("Agent", agent_world),
    ("Restored FS157", restored_world),
):
    ax.plot(
        arr[:, 0], arr[:, 1], arr[:, 2],
        label=name
    )

ax.set_xlabel("World X (m)")
ax.set_ylabel("World Y (m)")
ax.set_zlabel("World Z (m)")
ax.set_title("Reconstructed wrist trajectories")
ax.legend()

fig.tight_layout()
fig.savefig(OUT / "xyz_3d_world.png", dpi=160)
plt.close(fig)


# -------------------------------------------------
# Raw Camera 1 reprojection overlay
# -------------------------------------------------

cap = cv2.VideoCapture(str(TAKE / "camera1.avi"))

if not cap.isOpened():
    raise RuntimeError("Cannot open camera1.avi")

fps = cap.get(cv2.CAP_PROP_FPS) or FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_path = OUT / "trajectory_reprojection_camera1.mp4"

writer = cv2.VideoWriter(
    str(video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Cannot open video writer")


def draw(img, uv, color, label, radius=8):
    if uv is None or not np.isfinite(uv).all():
        return

    x, y = np.rint(uv).astype(int)

    if not (0 <= x < width and 0 <= y < height):
        return

    cv2.circle(img, (x, y), radius+2, (0, 0, 0), 3)
    cv2.circle(img, (x, y), radius, color, 2)

    cv2.putText(
        img, label,
        (min(width-180, x+12), max(115, y-12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, color, 2
    )


i = 0

while i < N:
    ok, img = cap.read()

    if not ok:
        break

    obs = observed_mask_raw(i)

    # White = measured image-space SAM3 mask centre
    draw(img, obs, (255, 255, 255), "SAM3 mask", 5)

    # Yellow = FS projected from original Camera 1 XYZ
    draw(
        img, project_table(fs[i]),
        (0, 255, 255), "FS projected", 9
    )

    # Magenta = agent accepted point
    draw(
        img, project_table(agent[i]),
        (255, 0, 255), "Agent projected", 7
    )

    # Green = restored FS157
    draw(
        img, project_table(restored[i]),
        (0, 255, 0), "Restored projected", 5
    )

    cv2.rectangle(
        img, (0, 0), (width, 100),
        (20, 20, 20), -1
    )

    cv2.putText(
        img,
        f"Frame {i}/{N-1} | FS={int(valid(fs)[i])} "
        f"Agent={int(valid(agent)[i])} "
        f"Restored={int(valid(restored)[i])}",
        (20, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (255, 255, 255), 2
    )

    cv2.putText(
        img,
        "WHITE: mask | YELLOW: FS | "
        "MAGENTA: agent | GREEN: restored",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62, (255, 255, 255), 2
    )

    writer.write(img)
    i += 1

cap.release()
writer.release()


# -------------------------------------------------
# Report
# -------------------------------------------------

def median_metric(key):
    values = np.array([
        float(r[key]) for r in diagnostics
    ])

    good = values[np.isfinite(values)]

    return float(np.median(good)) if len(good) else None


report = {
    "frames": N,
    "fs_valid": int(valid(fs).sum()),
    "agent_valid": int(valid(agent).sum()),
    "restored_valid": int(valid(restored).sum()),
    "median_fs_agent_disagreement_mm":
        median_metric("fs_agent_difference_mm"),
    "median_fs_restored_difference_mm":
        median_metric("fs_restored_difference_mm"),
    "median_fs_projection_mask_error_px":
        median_metric("fs_projection_mask_error_px"),
    "median_agent_projection_mask_error_px":
        median_metric("agent_projection_mask_error_px"),
    "median_restored_projection_mask_error_px":
        median_metric("restored_projection_mask_error_px"),
}

(OUT / "report.json").write_text(
    json.dumps(report, indent=2)
)

print("\n========== WRIST MAPPING DIAGNOSTIC ==========")

for k, v in report.items():
    print(f"{k}: {v}")

print("\nFiles saved to:", OUT)
print("XYZ table plot: xyz_table_comparison.png")
print("XYZ world plot: xyz_world_comparison.png")
print("Step plot: xyz_frame_jumps.png")
print("3D trajectory: xyz_3d_world.png")
print("Raw overlay: trajectory_reprojection_camera1.mp4")
print("Per-frame CSV: trajectory_mapping.csv")
