import cv2
import numpy as np
from pathlib import Path

BASE = Path("stereo/calibration")
CAL = BASE / "results_new/stereo_calibration.npz"
PAIRS = BASE / "pairs_new"

SQUARE = 0.02425
MARKER = 0.018

with np.load(CAL) as f:
    K1, D1 = f["K1"], f["D1"]
    K2, D2 = f["K2"], f["D2"]
    R, T = f["R"], f["T"].reshape(3, 1)

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

board = cv2.aruco.CharucoBoard(
    (7, 10), SQUARE, MARKER, dictionary
)
detector = cv2.aruco.CharucoDetector(board)

board_xyz = np.asarray(
    board.getChessboardCorners(), np.float64
)

P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = np.hstack([R, T])

all_length_errors = []
all_reprojection_errors = []
valid_pairs = 0
total_points = 0

for left_path in sorted(PAIRS.glob("left_*.png")):
    right_path = PAIRS / left_path.name.replace("left_", "right_")

    if not right_path.exists():
        continue

    a = cv2.imread(str(left_path))
    b = cv2.imread(str(right_path))

    if a is None or b is None:
        continue

    ca, ia, _, _ = detector.detectBoard(a)
    cb, ib, _, _ = detector.detectBoard(b)

    if any(x is None for x in (ca, ia, cb, ib)):
        continue

    da = {
        int(i): pt
        for i, pt in zip(
            np.asarray(ia).reshape(-1),
            np.asarray(ca).reshape(-1, 2)
        )
    }

    db = {
        int(i): pt
        for i, pt in zip(
            np.asarray(ib).reshape(-1),
            np.asarray(cb).reshape(-1, 2)
        )
    }

    ids = sorted(set(da) & set(db))

    if len(ids) < 8:
        continue

    p1 = np.asarray([da[i] for i in ids], np.float64)
    p2 = np.asarray([db[i] for i in ids], np.float64)

    u1 = cv2.undistortPoints(
        p1.reshape(-1, 1, 2), K1, D1
    ).reshape(-1, 2)

    u2 = cv2.undistortPoints(
        p2.reshape(-1, 1, 2), K2, D2
    ).reshape(-1, 2)

    h = cv2.triangulatePoints(P1, P2, u1.T, u2.T)

    if np.any(np.abs(h[3]) < 1e-12):
        continue

    xyz = (h[:3] / h[3]).T
    xyz2 = (R @ xyz.T + T).T

    valid = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 2] > 0)
        & (xyz2[:, 2] > 0)
    )

    if np.count_nonzero(valid) < 8:
        continue

    ids = np.asarray(ids)[valid]
    xyz = xyz[valid]
    p1 = p1[valid]
    p2 = p2[valid]

    valid_pairs += 1
    total_points += len(ids)

    # Validate adjacent horizontal and vertical board corners.
    # ChArUco grid contains (7-1) x (10-1) internal corners.
    mapping = {
        int(i): xyz[j]
        for j, i in enumerate(ids)
    }

    for i in ids:
        row, col = divmod(int(i), 6)

        for neighbor in (
            int(i) + 1 if col < 5 else -1,
            int(i) + 6 if row < 8 else -1
        ):
            if neighbor not in mapping:
                continue

            distance = np.linalg.norm(
                mapping[int(i)] - mapping[neighbor]
            )

            all_length_errors.append(
                (distance - SQUARE) * 1000
            )

    # Reproject triangulated 3D points into both cameras.
    projected1, _ = cv2.projectPoints(
        xyz, np.zeros(3), np.zeros(3), K1, D1
    )

    rv2, _ = cv2.Rodrigues(R)

    projected2, _ = cv2.projectPoints(
        xyz, rv2, T, K2, D2
    )

    e1 = np.linalg.norm(
        projected1.reshape(-1, 2) - p1, axis=1
    )

    e2 = np.linalg.norm(
        projected2.reshape(-1, 2) - p2, axis=1
    )

    all_reprojection_errors.extend(
        np.maximum(e1, e2).tolist()
    )

errors = np.asarray(all_length_errors)

print("\n===== NEW STEREO 3D VALIDATION =====")
print("Calibration:", CAL)
print("Valid stereo pairs:", valid_pairs)
print("Triangulated board corners:", total_points)
print("Adjacent distances checked:", len(errors))

if len(errors):
    print("\nExpected adjacent distance: 24.25 mm")
    print(
        "Mean signed distance error:",
        round(float(np.mean(errors)), 3), "mm"
    )
    print(
        "Median absolute distance error:",
        round(float(np.median(np.abs(errors))), 3), "mm"
    )
    print(
        "95th percentile absolute error:",
        round(float(np.percentile(np.abs(errors), 95)), 3), "mm"
    )

if all_reprojection_errors:
    e = np.asarray(all_reprojection_errors)
    print("\nReprojection error:")
    print("Median:", round(float(np.median(e)), 3), "px")
    print("95th percentile:", round(float(np.percentile(e, 95)), 3), "px")

print("\nEstimated baseline:",
      round(float(np.linalg.norm(T) * 1000), 2), "mm")
