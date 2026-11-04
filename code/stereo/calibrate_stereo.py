#!/usr/bin/env python3
"""Calibrate two cameras from stationary ChArUco image pairs.

Uses RAW, unflipped images saved by charuco_capture.py.
Run from anywhere: python stereo/calibrate_stereo.py
"""
from pathlib import Path
import argparse
import re
import sys
import cv2
import numpy as np

SQUARES_X, SQUARES_Y = 7, 10
SQUARE_LENGTH_M = 0.02425  # CHANGE if physical print measures differently
MARKER_LENGTH_M = 0.018  # CHANGE if physical print measures differently
DICTIONARY_ID = cv2.aruco.DICT_6X6_250
MIN_SHARED = 8
MIN_VIEW_CORNERS = 8


def get_pairs(folder):
    # Expected left_001.png / right_001.png; supports .jpg as well.
    files = list(folder.iterdir()) if folder.exists() else []
    pattern = re.compile(r'^(left|right)_(\d+)\.(png|jpe?g)$', re.IGNORECASE)
    found = {}
    for p in files:
        m = pattern.match(p.name)
        if m:
            found.setdefault(m.group(2), {})[m.group(1).lower()] = p
    return [(k, v['left'], v['right']) for k,v in sorted(found.items(), key=lambda kv:int(kv[0])) if 'left' in v and 'right' in v]


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary)
    return board, cv2.aruco.CharucoDetector(board)


def detect(detector, gray):
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or ids is None:
        return None
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    if len(corners) != len(ids) or len(np.unique(ids)) != len(ids):
        return None
    return {int(i): corners[j] for j, i in enumerate(ids)}


def compute_error(obj_views, img_views, rvecs, tvecs, K, D):
    ss, n = 0., 0
    for obj, img, rv, tv in zip(obj_views, img_views, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rv, tv, K, D)
        diff = projected.reshape(-1,2) - img.reshape(-1,2)
        ss += float(np.sum(diff**2)); n += len(diff)
    return np.sqrt(ss / max(n,1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', type=Path, default=Path(__file__).resolve().parent / 'calibration' / 'pairs')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'calibration' / 'results')
    ap.add_argument('--legacy', action='store_true', help='Use for ChArUco boards printed with pre-OpenCV-4.6 pattern conventions')
    args = ap.parse_args()
    board, detector = make_board()
    if args.legacy:
        board.setLegacyPattern(True)
        detector = cv2.aruco.CharucoDetector(board)
    board_corners = np.asarray(board.getChessboardCorners(), np.float32)
    pairs = get_pairs(args.pairs)
    if not pairs:
        sys.exit(f'No image pairs found in {args.pairs}. Expected left_001.png and right_001.png')
    views = [[], []]  # (object points, image points) per camera
    stereo_obj, stereo_img1, stereo_img2 = [], [], []
    image_size = None
    counts = []
    for label, left, right in pairs:
        a, b = cv2.imread(str(left)), cv2.imread(str(right))
        if a is None or b is None:
            print(f'SKIP {label}: unreadable image'); continue
        if a.shape[:2] != b.shape[:2]:
            print(f'SKIP {label}: different image sizes'); continue
        size = (a.shape[1], a.shape[0])
        if image_size is None: image_size = size
        if image_size != size:
            print(f'SKIP {label}: resolution differs from other pairs'); continue
        d1, d2 = detect(detector, cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)), detect(detector, cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
        if not d1 or not d2:
            print(f'SKIP {label}: no valid detection in both cameras'); continue
        for idx, detection in enumerate((d1, d2)):
            if len(detection) >= MIN_VIEW_CORNERS:
                ids = sorted(detection)
                obj = board_corners[ids].reshape(-1,1,3).copy()
                img = np.array([detection[i] for i in ids], np.float32).reshape(-1,1,2)
                views[idx].append((obj, img))
        common = sorted(set(d1).intersection(d2))
        if len(common) >= MIN_SHARED:
            stereo_obj.append(board_corners[common].reshape(-1,1,3).copy())
            stereo_img1.append(np.array([d1[i] for i in common], np.float32).reshape(-1,1,2))
            stereo_img2.append(np.array([d2[i] for i in common], np.float32).reshape(-1,1,2))
            counts.append((len(d1), len(d2), len(common)))
            print(f'PAIR {label}: left={len(d1)}, right={len(d2)}, SHARED={len(common)}')
        else:
            print(f'SKIP stereo {label}: left={len(d1)}, right={len(d2)}, shared={len(common)}')
    if any(len(v) < 12 for v in views) or len(stereo_obj) < 12:
        sys.exit(f'Insufficient usable variety: cam1={len(views[0])}, cam2={len(views[1])}, stereo={len(stereo_obj)}. Capture at least 15-20 varied good pairs.')
    if len(stereo_obj) < 20:
        print('WARNING: fewer than 20 stereo pairs. Prefer 30-40 diverse pairs.')
    intrinsic = []
    flags = 0
    for idx, v in enumerate(views):
        objects, images = zip(*v)
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(list(objects), list(images), image_size, None, None, flags=flags)
        err = compute_error(objects, images, rvecs, tvecs, K, D)
        intrinsic.append((K, D, rms, err))
        print(f'Camera {idx+1}: {len(v)} views, RMS={rms:.3f}px, reprojection={err:.3f}px')
        print('K=\n', K, '\nDistortion=', D.ravel())
    (K1, D1, rms1, err1), (K2, D2, rms2, err2) = intrinsic
    stereo_rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        stereo_obj, stereo_img1, stereo_img2, K1, D1, K2, D2,
        image_size, flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 150, 1e-8))
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T)
    baseline_mm = float(np.linalg.norm(T) * 1000.)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / 'stereo_calibration.npz', K1=K1, D1=D1, K2=K2, D2=D2,
             R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             image_size=np.array(image_size), square_length_m=SQUARE_LENGTH_M,
             marker_length_m=MARKER_LENGTH_M, stereo_rms=stereo_rms,
             camera1_rms=rms1, camera2_rms=rms2, baseline_mm=baseline_mm,
             legacy_pattern=args.legacy)
    np.savez(args.out / 'cam1_intrinsics.npz', K=K1, D=D1, image_size=image_size, rms=rms1)
    np.savez(args.out / 'cam2_intrinsics.npz', K=K2, D=D2, image_size=image_size, rms=rms2)
    print('\n===== STEREO CALIBRATION =====')
    print('Image size:', image_size)
    print('Usable stereo pairs:', len(stereo_obj))
    print('Average shared corners:', round(float(np.mean([x[2] for x in counts])), 1))
    print(f'Stereo RMS: {stereo_rms:.3f} pixels')
    print(f'Estimated 3D baseline: {baseline_mm:.1f} mm')
    print('T (metres):', T.ravel())
    print('Results saved in:', args.out)
    print('NOTE: Do NOT use this calibration until residuals, rectification and a known-distance target have been checked.')

if __name__ == '__main__': main()
