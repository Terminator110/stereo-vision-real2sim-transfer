import csv
from pathlib import Path
import numpy as np

ROOT = Path.home() / "physikal_workflow/dex-5"
TAKE = "take_20261103_211944_200460"

SRC = ROOT / f"FoundationStereo/wristband_smoothed_{TAKE}/wrist_xyz_smoothed.csv"
PALM = ROOT / f"outputs/reconstructed_demos/{TAKE}/trial/palm_pose_6dof.csv"
CALIB = ROOT / "code/stereo/calibration/parallel_results_v3/table_frame_multi.npz"

OUT = ROOT / f"outputs/smoothed_186_sim/{TAKE}"
OUT.mkdir(parents=True, exist_ok=True)

with np.load(CALIB) as z:
    R = z["R_cam1_to_table"].astype(float)
    t = z["t_cam1_to_table"].reshape(3).astype(float)

with SRC.open(newline="") as f:
    wrist = list(csv.DictReader(f))

with PALM.open(newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    palms = list(reader)

assert len(wrist) == len(palms) == 186

wrist_by_frame = {
    int(r["source_video_frame"]): r
    for r in wrist
}

quality_rows = []

for row in palms:
    frame = int(row["source_video_frame"])
    w = wrist_by_frame[frame]

    cam = np.array([
        float(w["smoothed_X_cam1_m"]),
        float(w["smoothed_Y_cam1_m"]),
        float(w["smoothed_Z_cam1_m"]),
    ])

    assert np.isfinite(cam).all(), f"Invalid XYZ at frame {frame}"

    table = R @ cam + t

    row["X_table_m"] = str(table[0])
    row["Y_table_m"] = str(table[1])
    row["Z_table_m"] = str(table[2])

    # Keep existing MediaPipe orientation and palm_status unchanged.
    row["wristband_depth_status"] = "VALID"

    quality_rows.append({
        "frame": frame,
        "source": w["source"],
        "smoothed_depth_m": cam[2],
    })

output = OUT / "palm_pose_6dof_smoothed.csv"

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(palms)

with (OUT / "wrist_source_quality.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["frame", "source", "smoothed_depth_m"]
    )
    writer.writeheader()
    writer.writerows(quality_rows)

print("========== SMOOTHED 186 SIM READY ==========")
print("Frames:", len(palms))
print("Palm orientations preserved")
print("Finger targets unchanged")
print("Output:", output)
print("Quality:", OUT / "wrist_source_quality.csv")
