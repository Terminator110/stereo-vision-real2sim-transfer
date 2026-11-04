#!/usr/bin/env python3
"""Experimental markerless stereo 3D wrist tracker using a palm-anchor model.

Coordinates: first (LEFT) calibration camera optical frame, in the units of stereo T.
NOT a clinical anatomical wrist joint centre: MediaPipe landmark 0 is a surface proxy.
Camera order, resolution, and orientation MUST match the calibration.
"""
import argparse
import csv
import os
import time
import cv2
import numpy as np
import mediapipe as mp

# MCP joints and nearby palm joints; avoid highly articulating distal fingertips.
ANCHORS = (1, 2, 5, 9, 13, 17)
TRACK_IDS = (0,) + ANCHORS
EDGES = ((0, 5), (5, 9), (9, 13), (13, 17), (0, 17), (1, 2))


def load_calibration(path):
    if path.lower().endswith('.npz'):
        with np.load(path, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
    else:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f'Cannot open calibration: {path}')
        data = {}
        for key in ('K1', 'D1', 'K2', 'D2', 'R', 'T', 'cameraMatrix1',
                    'cameraMatrix2', 'distCoeffs1', 'distCoeffs2', 'R_stereo', 'T_stereo'):
            node = fs.getNode(key)
            if not node.empty():
                data[key] = node.mat()
        fs.release()
    def get(*names):
        for name in names:
            if name in data:
                return np.asarray(data[name], np.float64)
        raise KeyError(f'Missing {names}. Available calibration keys: {list(data)}')
    k1 = get('K1', 'cameraMatrix1').reshape(3, 3)
    k2 = get('K2', 'cameraMatrix2').reshape(3, 3)
    d1 = get('D1', 'distCoeffs1').reshape(-1, 1)
    d2 = get('D2', 'distCoeffs2').reshape(-1, 1)
    r = get('R', 'R_stereo').reshape(3, 3)
    t = get('T', 'T_stereo').reshape(3, 1)
    if np.linalg.norm(t) < 1e-9 or abs(np.linalg.det(r) - 1) > .05:
        raise ValueError('Invalid stereo R or T')
    return k1, d1, k2, d2, r, t


def make_detector(model, confidence):
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence)
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect(det, frame, timestamp_ms):
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=im), timestamp_ms)
    if not out.hand_landmarks:
        return None, None
    h, w = frame.shape[:2]
    pts = np.array([(p.x * w, p.y * h) for p in out.hand_landmarks[0]], np.float64)
    handed = None
    if out.handedness:
        handed = out.handedness[0][0].category_name
    return pts, handed


def project(xyz, k, d, r=None, t=None):
    if r is None:
        r, t = np.eye(3), np.zeros((3, 1))
    rv, _ = cv2.Rodrigues(r)
    xy, _ = cv2.projectPoints(np.asarray(xyz, np.float64).reshape(-1, 3), rv,
                              np.asarray(t, np.float64).reshape(3, 1), k, d)
    return xy.reshape(-1, 2)


def triangulate(p0, p1, calib):
    k0, d0, k1, d1, r, t = calib
    a = cv2.undistortPoints(p0.reshape(-1, 1, 2), k0, d0).reshape(-1, 2)
    b = cv2.undistortPoints(p1.reshape(-1, 1, 2), k1, d1).reshape(-1, 2)
    P0 = np.column_stack([np.eye(3), np.zeros(3)])
    P1 = np.column_stack([r, t])
    hom = cv2.triangulatePoints(P0, P1, a.T, b.T)
    xyz = np.full((len(p0), 3), np.nan)
    validw = np.abs(hom[3]) > 1e-12
    xyz[validw] = (hom[:3, validw] / hom[3, validw]).T
    err = np.full(len(p0), np.inf)
    valid = validw & np.isfinite(xyz).all(axis=1)
    if valid.any():
        q0 = project(xyz[valid], k0, d0)
        q1 = project(xyz[valid], k1, d1, r, t)
        err[valid] = np.maximum(np.linalg.norm(q0 - p0[valid], axis=1),
                                np.linalg.norm(q1 - p1[valid], axis=1))
    z1 = (r @ xyz.T + t).T[:, 2]
    front = valid & (xyz[:, 2] > 0) & (z1 > 0)
    return xyz, err, front


def rigid_fit(source, target):
    """Return R,t,RMS mapping source -> target, without scaling."""
    a = source.mean(axis=0)
    b = target.mean(axis=0)
    x, y = source - a, target - b
    if np.linalg.matrix_rank(x, tol=1e-8) < 2:
        return None
    u, s, vt = np.linalg.svd(x.T @ y)
    flip = np.eye(3)
    flip[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = vt.T @ flip @ u.T
    trans = b - rot @ a
    residual = np.linalg.norm((rot @ source.T).T + trans - target, axis=1)
    return rot, trans, float(np.sqrt(np.mean(residual ** 2))), residual


def palm_model_estimate(template, xyz, good, max_residual):
    """Estimate wrist from an earlier stereo 3D palm pose using visible 3D anchors.

    Requires >=4 anchors; robustly excludes one bad anchor if it improves fit.
    Wrist displacement is based on rigid palm motion (assumption, not ground truth).
    """
    ids = [i for i in ANCHORS if good[i] and np.isfinite(template[i]).all()]
    if len(ids) < 4:
        return None, np.nan, 0
    src = template[ids]
    dst = xyz[ids]
    fit = rigid_fit(src, dst)
    if fit is None:
        return None, np.nan, 0
    if len(ids) >= 5:
        worst = int(np.argmax(fit[3]))
        reduced = [j for j in range(len(ids)) if j != worst]
        alt = rigid_fit(src[reduced], dst[reduced])
        if alt is not None and alt[2] < fit[2] * .65:
            fit = alt
            ids = [ids[j] for j in reduced]
    rot, trans, rms, _ = fit
    if rms > max_residual:
        return None, rms, len(ids)
    wrist = rot @ template[0] + trans
    if not np.isfinite(wrist).all() or wrist[2] <= 0:
        return None, rms, len(ids)
    return wrist, rms, len(ids)


def open_source(source, w, h):
    try:
        source = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera/video: {source}')
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f'Camera {source}: {actual}')
        if actual != (w, h):
            raise RuntimeError(f'Requested {(w, h)}, got {actual}')
    return cap


def annotate(frame, pts, label, wrist_model=None):
    if pts is not None:
        for i, j in EDGES:
            cv2.line(frame, tuple(np.rint(pts[i]).astype(int)),
                     tuple(np.rint(pts[j]).astype(int)), (80, 190, 80), 2)
        for i in TRACK_IDS:
            cv2.circle(frame, tuple(np.rint(pts[i]).astype(int)), 5,
                       (0, 0, 255) if i == 0 else (0, 220, 0), -1)
    if wrist_model is not None and np.isfinite(wrist_model).all():
        cv2.drawMarker(frame, tuple(np.rint(wrist_model).astype(int)), (255, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 0), 2)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--model', default='hand_landmarker.task')
    ap.add_argument('--left', default='2')
    ap.add_argument('--right', default='0')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--confidence', type=float, default=.40)
    ap.add_argument('--max-reproj', type=float, default=5.)
    ap.add_argument('--template-reproj', type=float, default=3.)
    ap.add_argument('--min-anchors', type=int, default=4)
    ap.add_argument('--max-palm-rms-ratio', type=float, default=.16,
                    help='Accept palm-fit RMS up to this fraction of stored palm width')
    ap.add_argument('--template-age', type=float, default=3.0,
                    help='Maximum seconds for rigid palm model recovery')
    ap.add_argument('--max-gap', type=float, default=.35)
    ap.add_argument('--alpha', type=float, default=.55)
    ap.add_argument('--max-speed', type=float, default=0.,
                    help='Optional continuity limit in calibration T units/second (0=disabled)')
    ap.add_argument('--output', default='tests/T14_mediapipe_multilandmark.csv')
    ap.add_argument('--flip-right-display', action='store_true',
                    help='Flip right PREVIEW vertically (never changes geometry)')
    ap.add_argument('--no-display', action='store_true')
    args = ap.parse_args()
    if not os.path.isfile(args.model):
        ap.error('MediaPipe model missing: ' + args.model)
    calib = load_calibration(args.calib)
    print('Stereo baseline:', np.linalg.norm(calib[5]), '(calibration T units)')
    cap0 = open_source(args.left, args.width, args.height)
    cap1 = open_source(args.right, args.width, args.height)
    det0 = make_detector(args.model, args.confidence)
    det1 = make_detector(args.model, args.confidence)
    fields = ('frame', 'time_s', 'mode', 'left_detected', 'right_detected', 'reason',
              'x', 'y', 'z', 'smooth_x', 'smooth_y', 'smooth_z', 'wrist_reproj_px',
              'anchor_good', 'palm_fit_rms', 'model_age_s', 'wrist_model_delta',
              'handedness_left', 'handedness_right')
    start = time.monotonic()
    frame_idx = 0
    template = None
    template_time = -np.inf
    palm_width = np.nan
    last = None
    last_at = -np.inf
    smooth = None
    smooth_at = -np.inf
    try:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            while cap0.grab() and cap1.grab():
                ok0, a = cap0.retrieve()
                ok1, b = cap1.retrieve()
                if not (ok0 and ok1):
                    break
                if a.shape[:2] != (args.height, args.width) or b.shape[:2] != (args.height, args.width):
                    raise RuntimeError('Frame dimensions mismatch calibration assumptions')
                now = time.monotonic()
                elapsed = now - start
                stamp = max(1, int(elapsed * 1000))
                p0, hand0 = detect(det0, a, stamp)
                p1, hand1 = detect(det1, b, stamp)
                row = dict.fromkeys(fields, '')
                row.update(frame=frame_idx, time_s=f'{elapsed:.4f}',
                           left_detected=int(p0 is not None), right_detected=int(p1 is not None),
                           handedness_left=hand0 or '', handedness_right=hand1 or '')
                wrist = None
                model_wrist = None
                mode = 'INVALID'
                reason = 'neither_detected'
                if p0 is None and p1 is not None:
                    reason = 'left_not_detected'
                elif p0 is not None and p1 is None:
                    reason = 'right_not_detected'
                elif p0 is not None and p1 is not None:
                    xyz, err, front = triangulate(p0, p1, calib)
                    valid = front & (err <= args.max_reproj)
                    anchor_count = sum(bool(valid[i]) for i in ANCHORS)
                    row['anchor_good'] = anchor_count
                    wrist_ok = bool(valid[0])
                    row['wrist_reproj_px'] = f'{err[0]:.3f}' if np.isfinite(err[0]) else ''
                    model_age = now - template_time
                    if template is not None:
                        row['model_age_s'] = f'{model_age:.3f}'
                    if template is not None and model_age <= args.template_age:
                        max_rms = args.max_palm_rms_ratio * palm_width
                        model_wrist, rms, fitted = palm_model_estimate(template, xyz, valid, max_rms)
                        if np.isfinite(rms):
                            row['palm_fit_rms'] = f'{rms:.5f}'
                    if wrist_ok:
                        wrist = xyz[0].copy()
                        mode = 'STEREO_WRIST'
                        reason = 'ok_direct'
                        if model_wrist is not None:
                            delta = float(np.linalg.norm(wrist - model_wrist))
                            row['wrist_model_delta'] = f'{delta:.5f}'
                            # Do not average across inconsistent anatomical estimates.
                            if delta > .30 * palm_width:
                                wrist = None
                                mode = 'INVALID'
                                reason = 'wrist_palm_disagree'
                        # Capture a model only from a high-quality complete palm observation.
                        if (wrist is not None and err[0] <= args.template_reproj and
                                sum(bool(front[i] and err[i] <= args.template_reproj) for i in ANCHORS) >= 4):
                            width = float(np.linalg.norm(xyz[5] - xyz[17])) if (front[5] and front[17] and err[5] <= args.template_reproj and err[17] <= args.template_reproj) else np.nan
                            if width > 1e-6 and np.isfinite(width):
                                template = xyz.copy()
                                for i in ANCHORS:
                                    if not (front[i] and err[i] <= args.template_reproj):
                                        template[i] = np.nan
                                template_time = now
                                palm_width = width
                    elif model_wrist is not None and anchor_count >= args.min_anchors:
                        wrist = model_wrist
                        mode = 'STEREO_PALM'
                        reason = 'wrist_recovered_from_palm'
                    elif not front[0]:
                        reason = 'wrist_behind_camera'
                    elif np.isfinite(err[0]) and err[0] > args.max_reproj:
                        reason = 'wrist_reprojection_high_no_palm'
                    else:
                        reason = 'insufficient_valid_landmarks'
                # Optional physical-speed continuity gate, no interpolation.
                if wrist is not None and args.max_speed > 0 and last is not None:
                    dt = now - last_at
                    if 0 < dt < args.max_gap and np.linalg.norm(wrist - last) / dt > args.max_speed:
                        wrist = None
                        mode = 'INVALID'
                        reason = 'speed_outlier'
                if wrist is not None:
                    last, last_at = wrist.copy(), now
                    if smooth is None or now - smooth_at > args.max_gap:
                        smooth = wrist.copy()
                    else:
                        smooth = args.alpha * wrist + (1 - args.alpha) * smooth
                    smooth_at = now
                    for name, value in zip(('x', 'y', 'z'), wrist):
                        row[name] = f'{value:.6f}'
                    for name, value in zip(('smooth_x', 'smooth_y', 'smooth_z'), smooth):
                        row[name] = f'{value:.6f}'
                else:
                    smooth = None
                row['mode'] = mode
                row['reason'] = reason
                writer.writerow(row)
                if frame_idx % 30 == 0:
                    f.flush()
                    point = np.round(wrist, 3) if wrist is not None else ''
                    print(f'{frame_idx:05d} {mode:14s} {point} reason={reason} anchors={row["anchor_good"]}')
                if not args.no_display:
                    overlay0 = None
                    overlay1 = None
                    if model_wrist is not None:
                        overlay0 = project(model_wrist[None], calib[0], calib[1])[0]
                        overlay1 = project(model_wrist[None], calib[2], calib[3], calib[4], calib[5])[0]
                    annotate(a, p0, f'LEFT {mode}', overlay0)
                    annotate(b, p1, f'RIGHT {reason}', overlay1)
                    if args.flip_right_display:
                        b = cv2.flip(b, 0)
                    combined = np.hstack((a, b))
                    if combined.shape[1] > 1800:
                        ratio = 1800 / combined.shape[1]
                        combined = cv2.resize(combined, None, fx=ratio, fy=ratio)
                    cv2.imshow('MediaPipe multi-landmark stereo wrist (q=quit)', combined)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        frame_idx += 1
                        break
                frame_idx += 1
    finally:
        det0.close()
        det1.close()
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()
        print(f'Saved {frame_idx} frames to {args.output}')


if __name__ == '__main__':
    main()
