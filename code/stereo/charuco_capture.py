#!/usr/bin/env python3
"""Capture paired full-resolution ChArUco images for later stereo calibration.
Run from your dex-5/code directory; image pairs are saved under stereo/calibration/pairs/.
"""
from pathlib import Path
import argparse
import time
import cv2
import numpy as np

SQUARES_X, SQUARES_Y = 7, 10
SQUARE_LENGTH_M = 0.02425   # UPDATE if measured printed square differs
MARKER_LENGTH_M = 0.018   # UPDATE if measured printed marker differs
DICT = cv2.aruco.DICT_6X6_250
WIDTH, HEIGHT, FPS = 1280, 720, 30


def make_detector(legacy=False):
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary
    )
    if legacy:
        board.setLegacyPattern(True)
    detector = cv2.aruco.CharucoDetector(board)
    return detector


def open_cam(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open /dev/video{idx}')
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f'/dev/video{idx}: {got[0]} x {got[1]}')
    if got != (WIDTH, HEIGHT):
        cap.release()
        raise RuntimeError('Actual resolution differs from calibration setting')
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam1', type=int, default=0)
    parser.add_argument('--cam2', type=int, default=2)
    parser.add_argument('--legacy', action='store_true',
                        help='Use only if printed board follows old OpenCV layout')
    parser.add_argument('--out', type=Path, default=Path('stereo/calibration/pairs'))
    parser.add_argument('--min-corners', type=int, default=8)
    args = parser.parse_args()
    detector = make_detector(args.legacy)
    args.out.mkdir(parents=True, exist_ok=True)
    cap1 = cap2 = None
    count = len(list(args.out.glob('left_*.png')))
    try:
        cap1, cap2 = open_cam(args.cam1), open_cam(args.cam2)
        print('Move/tilt a RIGID ChArUco board; hold still before pressing S.')
        print('S = save good pair, Q = quit. Preview is half-size; saved images are full-resolution.')
        failures = 0
        while True:
            g1, g2 = cap1.grab(), cap2.grab()
            if not (g1 and g2):
                failures += 1
                print(f'Camera grab failed ({failures}/5); check USB connection')
                if failures >= 5:
                    break
                time.sleep(0.2)
                continue
            ok1, frame1 = cap1.retrieve()
            ok2, frame2 = cap2.retrieve()
            if not (ok1 and ok2):
                print('Frame retrieve failed')
                break

            # Camera 2 is physically inverted.
            frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
            
            failures = 0
            previews, detections, valid_ids = [], [], []
            for label, frame in [('LEFT', frame1), ('RIGHT', frame2)]:
                corners, ids, marker_corners, marker_ids = detector.detectBoard(frame)
                # Some OpenCV 5 builds may return incomplete ChArUco
                # detections. Never draw or save them unless counts agree.
                nc = 0 if corners is None else len(corners)
                ni = 0 if ids is None else len(ids)
                valid = nc > 0 and nc == ni
                if valid:
                    c = np.asarray(corners)
                    i = np.asarray(ids)
                    valid = (c.size == 2 * nc and i.size == ni
                             and np.isfinite(c).all())
                n = nc if valid else 0
                detections.append(n)
                valid_ids.append(set(int(x) for x in np.asarray(ids).reshape(-1)) if valid else set())
                view = frame.copy()
                if marker_ids is not None and len(marker_ids) == len(marker_corners):
                    cv2.aruco.drawDetectedMarkers(view, marker_corners, marker_ids)
                if valid:
                    # OpenCV 5.0.0 may reject drawDetectedCornersCharuco even
                    # when detectBoard returns matching arrays. Draw manually.
                    coords = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
                    corner_ids = np.asarray(ids).reshape(-1)
                    for (x, y), corner_id in zip(coords, corner_ids):
                        pt = (int(round(x)), int(round(y)))
                        cv2.circle(view, pt, 5, (0, 255, 0), -1)
                        cv2.putText(view, str(int(corner_id)), (pt[0]+7, pt[1]-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
                elif nc or ni:
                    print(f'{label}: invalid detection, corners={nc}, ids={ni}')
                previews.append(cv2.resize(view, (640, 360)))
            shared_ids = valid_ids[0] & valid_ids[1]
            n_shared = len(shared_ids)
            for j, label in enumerate(('LEFT', 'RIGHT')):
                color = (0, 255, 0) if detections[j] >= args.min_corners else (0, 165, 255)
                cv2.rectangle(previews[j], (0, 0), (440, 36), (0, 0, 0), -1)
                cv2.putText(previews[j], f'{label}: {detections[j]} ChArUco corners',
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            combined = cv2.hconcat(previews)
            cv2.putText(combined, f'S: save | Q: quit | shared corners: {n_shared} | saved pairs: {count}',
                        (14, 700//2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.imshow('Stereo ChArUco capture', combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            if key == ord('s'):
                if min(detections) < args.min_corners or n_shared < args.min_corners:
                    print(f'NOT SAVED: left/right={detections}, common IDs={n_shared} (need >= {args.min_corners}).')
                    continue
                count += 1
                left = args.out / f'left_{count:03d}.png'
                right = args.out / f'right_{count:03d}.png'
                cv2.imwrite(str(left), frame1)
                cv2.imwrite(str(right), frame2)
                print(f'SAVED pair {count:03d}, corners L/R={detections}, shared={n_shared} -> {args.out}')
    finally:
        if cap1 is not None: cap1.release()
        if cap2 is not None: cap2.release()
        cv2.destroyAllWindows()
    print(f'Finished. {count} image pairs in {args.out.resolve()}')

if __name__ == '__main__':
    main()
