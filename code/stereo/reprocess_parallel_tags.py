#!/usr/bin/env python3
"""Reconstruct AprilTag XYZ from previously recorded, already-oriented stereo AVIs.
Uses the same detect()/triangulate() methods as stereo_wrist_parallel.py.
Never rotates camera2 twice. Backs up previous tags_xyz.csv.
"""
import argparse
import csv
from pathlib import Path
from collections import Counter
from datetime import datetime
import cv2
import numpy as np
from stereo_wrist_parallel import detect, triangulate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--take', type=Path, required=True)
    p.add_argument('--calib', type=Path, default=Path('stereo/calibration/parallel_results/stereo_calibration.npz'))
    p.add_argument('--ids', type=int, nargs='+', default=[4, 5, 2])
    p.add_argument('--max-reproj', type=float, default=3.0)
    a = p.parse_args()
    with np.load(a.calib) as d:
        K1, D1, K2, D2, R, T = [np.asarray(d[k], dtype=np.float64) for k in ('K1', 'D1', 'K2', 'D2', 'R', 'T')]
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    c1 = cv2.VideoCapture(str(a.take / 'camera1.avi'))
    c2 = cv2.VideoCapture(str(a.take / 'camera2.avi'))
    if not c1.isOpened() or not c2.isOpened():
        raise SystemExit('Could not open both videos')
    times = {}
    tf = a.take / 'timestamps.csv'
    if tf.is_file():
        with tf.open(newline='') as f:
            for row in csv.DictReader(f):
                times[int(row['frame'])] = row['time_s']
    dest = a.take / 'tags_xyz.csv'
    if dest.exists():
        backup = a.take / f'tags_xyz_before_reprocess_{datetime.now():%Y%m%d_%H%M%S}.csv'
        dest.rename(backup)
        print('Backed up old CSV:', backup, flush=True)
    stats = Counter()
    cols = ['frame','time_s','tag_id','x_cam1_m','y_cam1_m','z_cam1_m','reproj_px','cam1_seen','cam2_seen','valid']
    tmp = a.take / 'tags_xyz_reprocessing.tmp'
    try:
        with tmp.open('w', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=cols)
            writer.writeheader()
            i = 0
            while True:
                ok1, frame1 = c1.read()
                ok2, frame2 = c2.read()
                if not (ok1 and ok2):
                    break
                # Camera2 is already rotated 180 degrees in the saved AVI.
                first = detect(detector, frame1, set(a.ids))
                second = detect(detector, frame2, set(a.ids))
                if first.keys() & second.keys():
                    stats['frames_with_shared_id'] += 1
                for tag_id in a.ids:
                    row = dict(frame=i, time_s=times.get(i, ''), tag_id=tag_id,
                               x_cam1_m='', y_cam1_m='', z_cam1_m='', reproj_px='',
                               cam1_seen=int(tag_id in first), cam2_seen=int(tag_id in second), valid=0)
                    if tag_id in first and tag_id in second:
                        stats['shared_observations'] += 1
                        xyz, err = triangulate(first[tag_id], second[tag_id], K1,D1,K2,D2,R,T)
                        row['reproj_px'] = f'{err:.5f}'
                        if xyz is not None and np.isfinite(err) and err <= a.max_reproj:
                            row.update(x_cam1_m=f'{xyz[0]:.8f}', y_cam1_m=f'{xyz[1]:.8f}',
                                       z_cam1_m=f'{xyz[2]:.8f}', valid=1)
                            stats['valid_observations'] += 1
                            stats[f'valid_tag_{tag_id}'] += 1
                        else:
                            stats['geometry_rejected'] += 1
                    writer.writerow(row)
                i += 1
                if i % 50 == 0: print('Processed frames:', i, flush=True)
            stats['frames'] = i
        tmp.replace(dest)
    finally:
        c1.release(); c2.release()
    print('\nREPROCESSING RESULTS')
    for name, value in stats.items():
        print(f'{name}: {value}')
    print('Saved:', dest)
    print('XYZ = Camera 1 frame, metres; tag centres, not common wrist point.')
    if stats['valid_observations'] == 0:
        print('WARNING: Still no valid 3D positions. Check camera order, frame rotation and calibration correspondence.')

if __name__ == '__main__':
    main()
