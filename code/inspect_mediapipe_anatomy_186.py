import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / "physikal_workflow/dex-5"
TAKE = "take_20261103_211944_200460"

NPZ = ROOT / f"outputs/reconstructed_demos/{TAKE}/stereo_retarget/stereo_landmarks_and_targets.npz"
CALIB = ROOT / "code/stereo/calibration/parallel_results_v3/table_frame_multi.npz"
VIDEO = ROOT / f"code/stereo/recordings_parallel_v3/{TAKE}/camera1.avi"

OUT = ROOT / f"outputs/mediapipe_anatomy_check/{TAKE}"
OUT.mkdir(parents=True, exist_ok=True)

with np.load(NPZ) as z:
    print("Saved landmark arrays:", [(k, z[k].shape) for k in z.files])
    points = z["points_table_m"].astype(float)
    valid = z["landmark_valid"].astype(bool)

with np.load(CALIB) as z:
    R = z["R_cam1_to_table"].astype(float)
    t = z["t_cam1_to_table"].reshape(3).astype(float)

with np.load(
    ROOT / "code/stereo/calibration/parallel_results_v3/stereo_calibration.npz"
) as z:
    K = z["K1"].astype(float)
    D = z["D1"].astype(float)

assert points.shape == (186, 21, 3), points.shape
assert valid.shape == (186, 21), valid.shape

cap = cv2.VideoCapture(str(VIDEO))
assert cap.isOpened(), VIDEO

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Reconstruction is approximately 10 FPS, not encoded AVI 20 FPS.
FPS = 10.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUT / "mediapipe_anatomy.mp4"),
    fourcc, FPS, (W, H)
)

zoom_writer = cv2.VideoWriter(
    str(OUT / "thumb_zoom.mp4"),
    fourcc, FPS, (640, 640)
)

assert writer.isOpened() and zoom_writer.isOpened()

# MediaPipe 21-landmark hand topology.
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

names = {
    0: "WRIST",
    1: "THUMB CMC",
    2: "THUMB MCP",
    3: "THUMB IP",
    4: "THUMB TIP",
    5: "INDEX MCP",
    8: "INDEX TIP",
    12: "MIDDLE TIP",
}

quality = []

for i in range(186):
    ok, img = cap.read()
    if not ok:
        raise RuntimeError(f"Video ended before frame {i}")

    p = points[i]
    v = valid[i] & np.isfinite(p).all(axis=1)

    # Table XYZ -> original Camera 1 optical XYZ.
    cam = (R.T @ (p - t).T).T

    uv = np.full((21, 2), np.nan)

    for j in range(21):
        if not v[j] or cam[j, 2] <= 0:
            continue

        projected, _ = cv2.projectPoints(
            cam[j].reshape(1, 1, 3),
            np.zeros(3), np.zeros(3),
            K, D
        )

        uv[j] = projected.reshape(2)

    def available(j):
        return v[j] and np.isfinite(uv[j]).all()

    def xy(j):
        return tuple(np.rint(uv[j]).astype(int))

    # Skeleton.
    for a, b in connections:
        if not available(a) or not available(b):
            continue

        color = (0, 255, 255) if a in (1,2,3,4) or b in (1,2,3,4) else (0, 220, 0)

        cv2.line(img, xy(a), xy(b), color, 2)

    for j in range(21):
        if not available(j):
            continue

        color = (0, 255, 255) if 1 <= j <= 4 else (0, 255, 0)

        cv2.circle(img, xy(j), 4, color, -1)

    for j in (1,2,3,4,8,12):
        if available(j):
            cv2.putText(
                img, names.get(j, str(j)),
                (xy(j)[0] + 8, xy(j)[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255,255,255), 2
            )

    thumb_index_mm = np.nan

    if v[4] and v[8]:
        thumb_index_mm = float(
            np.linalg.norm(p[4] - p[8]) * 1000
        )

        if available(4) and available(8):
            cv2.line(
                img, xy(4), xy(8),
                (255, 0, 255), 2
            )

    # Minimal annotation; don't obscure the hand.
    cv2.putText(
        img,
        f"Frame {i:03d} | Valid landmarks {v.sum()}/21",
        (15, H - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255,255,255), 2
    )

    distance_text = (
        f"Thumb-index 3D: {thumb_index_mm:.0f} mm"
        if np.isfinite(thumb_index_mm)
        else "Thumb-index 3D: INVALID"
    )

    cv2.putText(
        img, distance_text,
        (15, H - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255,255,255), 2
    )

    writer.write(img)

    # Thumb/index close-up centred on the detected fingers.
    usable = [
        uv[j] for j in [1,2,3,4,5,6,7,8]
        if available(j)
    ]

    zoom = np.zeros((640, 640, 3), dtype=np.uint8)

    if usable:
        pts = np.array(usable)
        cx, cy = np.mean(pts, axis=0)

        span = max(
            np.ptp(pts[:,0]),
            np.ptp(pts[:,1]),
            80
        )

        size = int(np.clip(span * 2.3, 120, 500))

        # Crop from already annotated image.
        x0 = int(round(cx - size / 2))
        y0 = int(round(cy - size / 2))

        crop = np.zeros((size, size, 3), dtype=np.uint8)

        sx0 = max(0, x0)
        sy0 = max(0, y0)
        sx1 = min(W, x0 + size)
        sy1 = min(H, y0 + size)

        if sx1 > sx0 and sy1 > sy0:
            crop[
                sy0-y0:sy1-y0,
                sx0-x0:sx1-x0
            ] = img[sy0:sy1, sx0:sx1]

        zoom = cv2.resize(crop, (640,640))

    cv2.putText(
        zoom, f"Frame {i:03d}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (255,255,255), 2
    )

    zoom_writer.write(zoom)

    quality.append([
        i,
        int(v.sum()),
        int(v[1:5].all()),
        int(v[5:9].all()),
        thumb_index_mm
    ])

cap.release()
writer.release()
zoom_writer.release()

with (OUT / "thumb_detection_quality.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "frame",
        "valid_landmarks",
        "thumb_all_valid",
        "index_all_valid",
        "thumb_index_distance_3d_mm"
    ])
    w.writerows(quality)

q = np.array(quality, dtype=float)

print("\n========== MEDIAPIPE ANATOMY ==========")
print("Frames:", len(quality))
print("Thumb fully valid:", int(q[:,2].sum()))
print("Index fully valid:", int(q[:,3].sum()))
print("Thumb-index distance available:", int(np.isfinite(q[:,4]).sum()))
print("Output:", OUT)
