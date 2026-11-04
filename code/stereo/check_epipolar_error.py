import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
calib = np.load(BASE / "calibration/results/stereo_calibration.npz")
pairs = BASE / "calibration/pairs"

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

board = cv2.aruco.CharucoBoard(
    (7, 10), 0.025, 0.018, dictionary
)

detector = cv2.aruco.CharucoDetector(board)

errors = []

for left in sorted(pairs.glob("left_*.png")):
    right = pairs / left.name.replace("left_", "right_")

    if not right.exists():
        continue

    img1 = cv2.imread(str(left), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(right), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        continue

    c1, ids1, _, _ = detector.detectBoard(img1)
    c2, ids2, _, _ = detector.detectBoard(img2)

    if c1 is None or c2 is None or ids1 is None or ids2 is None:
        continue

    d1 = dict(zip(ids1.flatten(), c1.reshape(-1, 2)))
    d2 = dict(zip(ids2.flatten(), c2.reshape(-1, 2)))

    common = sorted(set(d1) & set(d2))

    if len(common) < 8:
        continue

    pts1 = np.array([d1[i] for i in common], np.float64)
    pts2 = np.array([d2[i] for i in common], np.float64)

    rect1 = cv2.undistortPoints(
        pts1.reshape(-1, 1, 2),
        calib["K1"], calib["D1"],
        R=calib["R1"], P=calib["P1"]
    )

    rect2 = cv2.undistortPoints(
        pts2.reshape(-1, 1, 2),
        calib["K2"], calib["D2"],
        R=calib["R2"], P=calib["P2"]
    )

    dy = np.abs(
        rect1.reshape(-1, 2)[:, 1] -
        rect2.reshape(-1, 2)[:, 1]
    )

    errors.extend(dy.tolist())

    print(
        f"{left.stem}: "
        f"shared={len(common)}, "
        f"mean={np.mean(dy):.3f}px, "
        f"max={np.max(dy):.3f}px"
    )

if not errors:
    raise RuntimeError("No valid matching ChArUco corners found")

errors = np.array(errors)

print("\n===== EPIPOLAR ALIGNMENT =====")
print(f"Correspondences: {len(errors)}")
print(f"Mean error: {np.mean(errors):.3f} px")
print(f"Median error: {np.median(errors):.3f} px")
print(f"95th percentile: {np.percentile(errors, 95):.3f} px")
print(f"Maximum error: {np.max(errors):.3f} px")