#!/usr/bin/env python3
"""Live dual-camera AprilTag 36h11 tracking for PARALLEL stereo calibration.
Camera 1 = /dev/video2, Camera 2 = /dev/video0 rotated 180 deg BEFORE detection.
All reported 3D positions are in CAMERA 1 coordinates, in metres.
Each tag centre is reported separately; no uncalibrated wrist centre is inferred.
"""
import argparse
import csv
import time
from pathlib import Path
import cv2
import numpy as np


def open_cam(index, width, height, fps):
    c = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not c.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{index}")
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    c.set(cv2.CAP_PROP_FPS, fps)
    actual = (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if actual != (width, height):
        c.release()
        raise RuntimeError(f"Camera {index} returned {actual}; calibration requires {(width,height)}")
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return c


def detect(detector, frame, ids_of_interest):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if i in ids_of_interest:
                found[i] = c.reshape(4, 2).astype(np.float64)
    return found


def triangulate(c1, c2, K1, D1, K2, D2, R, T):
    # Undistort to normalized camera coordinates, without image rectification.
    u1 = cv2.undistortPoints(c1.reshape(-1, 1, 2), K1, D1).reshape(-1, 2)
    u2 = cv2.undistortPoints(c2.reshape(-1, 1, 2), K2, D2).reshape(-1, 2)
    p1 = np.column_stack([np.eye(3), np.zeros(3)])
    p2 = np.column_stack([R, T.reshape(3)])
    homogeneous = cv2.triangulatePoints(p1, p2, u1.T, u2.T)
    xyz = (homogeneous[:3] / homogeneous[3]).T
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
        return None, float('inf')
    xyz2 = (R @ xyz.T + T.reshape(3, 1)).T
    if np.any(xyz2[:, 2] <= 0):
        return None, float('inf')
    p1_hat, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K1, D1)
    rotation_vector, _ = cv2.Rodrigues(R)
    p2_hat, _ = cv2.projectPoints(xyz, rotation_vector, T.reshape(3), K2, D2)
    error = np.concatenate([
        np.linalg.norm(p1_hat.reshape(-1, 2) - c1, axis=1),
        np.linalg.norm(p2_hat.reshape(-1, 2) - c2, axis=1)
    ])
    return xyz.mean(axis=0), float(np.sqrt(np.mean(error**2)))


def draw(frame, observed, label):
    cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    for i, pts in observed.items():
        pts_i = np.rint(pts).astype(np.int32)
        cv2.polylines(frame, [pts_i], True, (255, 0, 255), 2)
        centre = tuple(np.rint(pts.mean(axis=0)).astype(int))
        cv2.putText(frame, f"ID {i}", centre, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam1', type=int, default=2)
    ap.add_argument('--cam2', type=int, default=0)
    ap.add_argument('--rotate-cam2', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--calib', type=Path, default=Path('stereo/calibration/parallel_results/stereo_calibration.npz'))
    ap.add_argument('--out', type=Path, default=Path('stereo/parallel_wrist_tracking.csv'))
    ap.add_argument('--ids', type=int, nargs='+', default=[4, 5])
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--max-reproj', type=float, default=3.0)
    args = ap.parse_args()

    with np.load(args.calib) as d:
        K1, D1, K2, D2, R, T = [d[k].astype(np.float64) for k in ('K1','D1','K2','D2','R','T')]
        if 'image_size' in d:
            cal_size = tuple(int(x) for x in d['image_size'].reshape(-1))
            if cal_size != (args.width, args.height):
                raise RuntimeError(f"Calibration image size {cal_size} != requested {(args.width, args.height)}")
    print(f"Baseline: {np.linalg.norm(T)*1000:.1f} mm; XYZ frame: Camera 1")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ['frame','time_s','tag_id','cam1_seen','cam2_seen','valid','x_cam1_m','y_cam1_m','z_cam1_m','reproj_rms_px','reason']
    cam1 = cam2 = None
    counts = {'frames':0,'valid':0,'no_shared':0,'geometry_rejected':0}
    try:
        cam1 = open_cam(args.cam1, args.width, args.height, args.fps)
        cam2 = open_cam(args.cam2, args.width, args.height, args.fps)
        with args.out.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            t0 = time.monotonic()
            while True:
                ok1 = cam1.grab()
                ok2 = cam2.grab()
                if not (ok1 and ok2):
                    print('Camera grab failed'); break
                ok1, frame1 = cam1.retrieve()
                ok2, frame2 = cam2.retrieve()
                if not (ok1 and ok2):
                    print('Camera retrieve failed'); break
                if args.rotate_cam2:
                    frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
                elapsed = time.monotonic() - t0
                first = detect(detector, frame1, set(args.ids))
                second = detect(detector, frame2, set(args.ids))
                n = counts['frames']
                shared = first.keys() & second.keys()
                if not shared:
                    counts['no_shared'] += 1
                for tag_id in args.ids:
                    has1 = tag_id in first
                    has2 = tag_id in second
                    row = dict(frame=n, time_s=f'{elapsed:.6f}', tag_id=tag_id,
                               cam1_seen=int(has1), cam2_seen=int(has2), valid=0,
                               x_cam1_m='', y_cam1_m='', z_cam1_m='', reproj_rms_px='', reason='')
                    if has1 and has2:
                        xyz, err = triangulate(first[tag_id], second[tag_id], K1,D1,K2,D2,R,T)
                        row['reproj_rms_px'] = f'{err:.4f}'
                        if xyz is not None and err <= args.max_reproj:
                            row.update(valid=1, x_cam1_m=f'{xyz[0]:.6f}',
                                       y_cam1_m=f'{xyz[1]:.6f}', z_cam1_m=f'{xyz[2]:.6f}', reason='ok')
                            counts['valid'] += 1
                        else:
                            row['reason'] = 'geometry_rejected'
                            counts['geometry_rejected'] += 1
                    else:
                        row['reason'] = 'no_shared_id'
                    writer.writerow(row)
                counts['frames'] += 1
                if n % 30 == 0:
                    f.flush()
                draw(frame1, first, f'CAM1 /dev/video{args.cam1}')
                draw(frame2, second, f'CAM2 /dev/video{args.cam2} ROTATED 180' if args.rotate_cam2 else 'CAM2')
                view = cv2.hconcat([cv2.resize(frame1, (640,360)), cv2.resize(frame2, (640,360))])
                cv2.putText(view, f'Frame {n} | Stereo valid tag observations: {counts["valid"]} | Q=quit',
                            (15, 705//2), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
                cv2.imshow('Parallel stereo wrist tags 4+5', view)
                if cv2.waitKey(1) & 0xFF in (ord('q'),27):
                    break
    finally:
        if cam1: cam1.release()
        if cam2: cam2.release()
        cv2.destroyAllWindows()
    print(f"Frames={counts['frames']}, valid tag observations={counts['valid']}, "
          f"frames with no shared ID={counts['no_shared']}, geometry rejected={counts['geometry_rejected']}")
    print(f"Saved CSV: {args.out}")
    print('Important: CSV rows are tag centres, NOT anatomical wrist centre. Old table transform is invalid.')


if __name__ == '__main__':
    main()
