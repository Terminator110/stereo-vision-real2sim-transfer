import csv
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

ROOT = Path.home() / "physikal_workflow/dex-5"

TAKE_ID = (
    sys.argv[1] if len(sys.argv) > 1
    else "take_20261103_211944_200460"
)

trial = ROOT / "FoundationStereo" / f"wristband_batch_{TAKE_ID}"

take = (
    ROOT / "code/stereo/recordings_parallel_v3" / TAKE_ID
)

wrist_csv = trial / "wrist_xyz_camera1.csv"
quality_csv = trial / "stereo_retarget/stereo_quality.csv"

if not wrist_csv.exists():
    raise SystemExit(f"Missing: {wrist_csv}")

with wrist_csv.open() as f:
    wrist = list(csv.DictReader(f))

print("\n========== WRIST XYZ ==========")

status = Counter(r["status"] for r in wrist)
print("Total frames:", len(wrist))
print("Status:", dict(status))

bad = [
    int(r["source_video_frame"])
    for r in wrist if r["status"] != "VALID"
]

print("Invalid wrist frames:", len(bad))

runs = []
for i in bad:
    if runs and i == runs[-1][-1] + 1:
        runs[-1].append(i)
    else:
        runs.append([i])

print("\nInvalid intervals:")
for r in runs:
    print(f"  {r[0]:3d}-{r[-1]:3d}: {len(r)} frames")

print("\n========== FINGER RETARGETING ==========")

if quality_csv.exists():
    with quality_csv.open() as f:
        q = list(csv.DictReader(f))

    joints = [
        k for k in q[0]
        if k.startswith(("Yaw_", "Pitch_", "Roll_"))
    ]

    print("Stereo frames:", len(q))
    print("Finger joints:", len(joints))

    print("\nFinger states:")
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        key = finger + "_state"
        if key in q[0]:
            print(finger, dict(Counter(r[key] for r in q)))

    print("\nFinger joint movement (degrees):")
    for joint in joints:
        values = []
        for r in q:
            try:
                values.append(float(r[joint]))
            except (ValueError, TypeError):
                values.append(float("nan"))

        arr = np.asarray(values)
        arr = arr[np.isfinite(arr)]

        if len(arr):
            movement = np.degrees(np.ptp(arr))
            print(f"  {joint:12s} {movement:8.2f}")

    print("\nThumb yaw column:",
          "PRESENT" if "Yaw_11R" in joints else "MISSING")
else:
    print("Stereo quality CSV not found.")

print("\n========== CREATE INSPECTION CONTACT SHEET ==========")

video = take / "camera1.avi"
cap = cv2.VideoCapture(str(video))

if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {video}")

# Select representative frames from both valid and invalid regions.
valid_frames = [
    int(r["source_video_frame"])
    for r in wrist if r["status"] == "VALID"
]

selected = set()

for group in (bad, valid_frames):
    if group:
        indices = np.linspace(
            0, len(group)-1,
            min(12, len(group)),
            dtype=int
        )
        selected.update(group[i] for i in indices)

selected = sorted(selected)

tiles = []

for frame in selected:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()

    if not ok:
        continue

    img = cv2.resize(img, (320, 180))

    label = "INVALID WRIST" if frame in bad else "VALID WRIST"

    color = (
        (0, 0, 255) if frame in bad
        else (0, 255, 0)
    )

    cv2.putText(
        img,
        f"FRAME {frame}: {label}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2
    )

    tiles.append(img)

cap.release()

if tiles:
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))

    sheet = np.vstack([
        np.hstack(tiles[i:i+4])
        for i in range(0, len(tiles), 4)
    ])

    output = trial / "wrist_finger_inspection.jpg"
    cv2.imwrite(str(output), sheet)
    print("Contact sheet:", output)

print("\n========== DONE ==========")
