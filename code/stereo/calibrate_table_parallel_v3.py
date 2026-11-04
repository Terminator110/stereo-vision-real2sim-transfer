#!/usr/bin/env python3
"""Calibrate tabletop plane and a *provisional* table coordinate frame using AprilTag ID 3.
Camera 1 /dev/video2, camera 2 /dev/video0 rotated once by 180 degrees.
Move a rigid flat tag to 10+ positions on the same tabletop plane, press S at each.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cam1', type=int, default=2)
    ap.add_argument('--cam2', type=int, default=0)
    ap.add_argument('--rotate-cam2', action='store_true')
    ap.add_argument('--calib', type=Path, default=Path('stereo/calibration/parallel_results_v3/stereo_calibration.npz'))
    ap.add_argument('--out', type=Path, default=Path('stereo/calibration/parallel_results_v3/table_frame_multi.npz'))
    ap.add_argument('--tag-id', type=int, default=3)
    ap.add_argument('--min-positions', type=int, default=10)
    args = ap.parse_args()

    cal = np.load(args.calib)
    K1, D1, K2, D2, R, T = (cal[k] for k in ('K1', 'D1', 'K2', 'D2', 'R', 'T'))
    T = np.asarray(T, dtype=float).reshape(3, 1)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    def open_cam(idx):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open /dev/video{idx}')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual != (1280, 720):
            cap.release()
            raise RuntimeError(f'/dev/video{idx}: {actual} != 1280x720')
        return cap

    cap1, cap2 = None, None
    points = []
    samples = []

    def tag_corners(frame):
        corners, ids, _ = detector.detectMarkers(frame)
        out = {}
        if ids is not None:
            for i, tag_id in enumerate(ids.flatten()):
                out[int(tag_id)] = np.asarray(corners[i], dtype=float).reshape(4, 2)
        return out

    def triangulate(c1, c2):
        u1 = cv2.undistortPoints(np.asarray(c1).reshape(-1, 1, 2), K1, D1).reshape(-1, 2).T
        u2 = cv2.undistortPoints(np.asarray(c2).reshape(-1, 1, 2), K2, D2).reshape(-1, 2).T
        hom = cv2.triangulatePoints(P1, P2, u1, u2)
        pts = (hom[:3, :] / hom[3:4, :]).T
        return pts

    try:
        cap1, cap2 = open_cam(args.cam1), open_cam(args.cam2)
        print('Place AprilTag flat on TABLE surface. Move across positions, keep FLAT, press S. Q to finish.')
        print('IMPORTANT: tag must be on the table, not held above it; no camera movement.')
        print('Camera2 rotate once:', args.rotate_cam2)
        while True:
            a, b = cap1.grab(), cap2.grab()
            if not (a and b):
                print('Camera grab failed')
                break
            a, f1 = cap1.retrieve()
            b, f2 = cap2.retrieve()
            if not (a and b):
                print('Camera retrieve failed')
                break
            if args.rotate_cam2:
                f2 = cv2.rotate(f2, cv2.ROTATE_180)
            t1, t2 = tag_corners(f1), tag_corners(f2)
            shared = args.tag_id in t1 and args.tag_id in t2
            for f, tags in ((f1, t1), (f2, t2)):
                for tag_id, cs in tags.items():
                    if tag_id == args.tag_id:
                        cv2.polylines(f, [cs.astype(np.int32)], True, (0, 255, 0), 2)
                        c = np.mean(cs, axis=0).astype(int)
                        cv2.putText(f, str(tag_id), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            preview = cv2.hconcat([cv2.resize(f1, (640, 360)), cv2.resize(f2, (640, 360))])
            message = f'ID {args.tag_id}: {"BOTH VIEWS" if shared else "NOT BOTH VIEWS"} | saved={len(samples)} | S=save Q=quit'
            cv2.putText(preview, message, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 255, 0) if shared else (0, 0, 255), 2)
            cv2.imshow('Table calibration V3', preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                if not shared:
                    print('NOT SAVED: tag not detected in both cameras')
                    continue
                p = triangulate(t1[args.tag_id], t2[args.tag_id])
                if not np.isfinite(p).all() or np.any(p[:, 2] <= 0):
                    print('NOT SAVED: invalid triangulation')
                    continue
                # Reject stereo disparity incompatible with calibration using pixel reprojection RMS.
                q1, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K1, D1)
                q2, _ = cv2.projectPoints(p, cv2.Rodrigues(R)[0], T, K2, D2)
                err = np.sqrt(np.mean(np.concatenate([(q1.reshape(-1,2)-t1[args.tag_id]).ravel(), (q2.reshape(-1,2)-t2[args.tag_id]).ravel()])**2))
                if err > 3.0:
                    print(f'NOT SAVED: reproj RMS {err:.2f}px > 3px')
                    continue
                samples.append(p.copy())
                points.extend(p)
                print(f'SAVED position {len(samples)}, centre cam1(mm)={np.mean(p,axis=0)*1000}, reproj={err:.2f}px')
    finally:
        if cap1 is not None: cap1.release()
        if cap2 is not None: cap2.release()
        cv2.destroyAllWindows()

    if len(samples) < args.min_positions:
        raise SystemExit(f'Only {len(samples)} positions. Need >= {args.min_positions}; no output saved.')
    pts = np.asarray(points)
    centroid = pts.mean(axis=0)
    _u, _s, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vh[-1].copy()
    # Camera 1 is physically above the tabletop; normal points toward camera 1.
    if np.dot(normal, -centroid) < 0: normal = -normal
    distance = (pts - centroid) @ normal
    rms_mm = float(np.sqrt(np.mean(distance**2))*1000)
    p95_mm = float(np.percentile(np.abs(distance),95)*1000)

    # Provisional origin = first sampled tag centre projected to plane.
    # Provisional X = first tag's corner0->corner1 direction, projected onto plane.
    first = samples[0]
    origin = first.mean(axis=0)
    origin = origin - np.dot(origin-centroid, normal)*normal
    x_dir = (first[1]-first[0]).copy()
    x_dir -= np.dot(x_dir, normal)*normal
    norm = np.linalg.norm(x_dir)
    if norm < 1e-5: raise RuntimeError('Cannot establish X from first tag edge')
    x_dir /= norm
    y_dir = np.cross(normal, x_dir)
    y_dir /= np.linalg.norm(y_dir)
    R_cam1_to_table = np.vstack([x_dir, y_dir, normal])
    t_cam1_to_table = -R_cam1_to_table @ origin
    R_table_to_cam1 = R_cam1_to_table.T
    t_table_to_cam1 = origin
    R_table_to_cam2 = R @ R_table_to_cam1
    t_table_to_cam2 = R @ t_table_to_cam1 + T.flatten()
    cam1_in_table = t_cam1_to_table
    cam2_in_table = R_cam1_to_table @ (-R.T @ T.flatten()) + t_cam1_to_table

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
        R_cam1_to_table=R_cam1_to_table, t_cam1_to_table=t_cam1_to_table,
        R_table_to_cam1=R_table_to_cam1, t_table_to_cam1=t_table_to_cam1,
        R_table_to_cam2=R_table_to_cam2, t_table_to_cam2=t_table_to_cam2,
        camera1_in_table=cam1_in_table, camera2_in_table=cam2_in_table,
        table_normal_cam1=normal, table_origin_cam1_m=origin,
        plane_rms_mm=rms_mm, plane_p95_mm=p95_mm,
        number_of_positions=len(samples), tag_id=args.tag_id,
        points_cam1_m=pts)
    print(f'\nSaved: {args.out}')
    print(f'Positions: {len(samples)} | plane RMS: {rms_mm:.2f}mm | p95: {p95_mm:.2f}mm')
    print('PROVISIONAL TABLE FRAME: origin=first tag centre; X=first tag edge; Z=up.')
    print('To align to real table edges / MuJoCo, a fixed origin and X axis reference are STILL needed.')

if __name__ == '__main__':
    main()
