"""Stereo-only hand shape and wrist orientation for stationary G1/Dex5.

MediaPipe supplies 2D image landmarks only. All control geometry is reconstructed
from BOTH calibrated views. No monocular 3D fallback or arm/XYZ retargeting.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import threading
import time

import cv2
import mediapipe as mp
import mujoco
import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g1 = load_local('_stereo_g1_existing', CODE/'10_g1_dex5_stationary.py')
stereo = load_local('_stereo_wrist_existing', CODE/'stereo/stereo_wrist_test.py')
PALM = (0, 5, 9, 17)
GROUP_POINTS = {
    'wrist': PALM,
    'thumb': (0, 1, 2, 3, 4, 5, 9, 17),
    **{name: tuple(sorted({0, 5, 17, *chain})) for name, chain in g1.retarget.CHAINS.items()},
}
CALIBRATION_MAX_MOTION_DEG = 8.0


def check_rotation(matrix, name):
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f'Invalid {name}')
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(matrix), 1, atol=1e-5):
        raise ValueError(f'{name} is not a proper rotation')


def load_calibration(calibration_path, table_path):
    with np.load(calibration_path, allow_pickle=False) as saved:
        cal = {key: np.array(saved[key], dtype=np.float64) for key in ('K1', 'D1', 'K2', 'D2', 'R', 'T')}
        raw_size = np.asarray(saved['image_size'], dtype=float)
        if raw_size.shape != (2,) or not np.isfinite(raw_size).all() or np.any(raw_size <= 0) or np.any(raw_size != np.rint(raw_size)):
            raise ValueError('Invalid calibration image_size')
        size = tuple(map(int, raw_size))
    for key in ('K1', 'K2'):
        matrix = cal[key]
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
            raise ValueError(f'Invalid camera intrinsics {key}')
        if not np.allclose(matrix[2], [0, 0, 1]):
            raise ValueError(f'Invalid intrinsic bottom row: {key}')
    for key in ('D1', 'D2'):
        if cal[key].size not in (4, 5, 8, 12, 14) or not np.isfinite(cal[key]).all():
            raise ValueError(f'Invalid distortion coefficients {key}')
    check_rotation(cal['R'], 'stereo R')
    cal['T'] = cal['T'].reshape(3, 1)
    if not np.isfinite(cal['T']).all() or np.linalg.norm(cal['T']) < 1e-6:
        raise ValueError('Invalid stereo baseline')
    with np.load(table_path, allow_pickle=False) as saved:
        rotation = np.array(saved['R_cam1_to_table'], dtype=float)
        translation = np.array(saved['t_cam1_to_table'], dtype=float).reshape(3)
        check_rotation(rotation, 'camera1-to-table R')
        if not np.isfinite(translation).all():
            raise ValueError('Invalid camera1-to-table translation')
        # Reject a table file whose saved inverse/camera2 pose disagrees with this calibration.
        inverse_r = np.array(saved['R_table_to_cam1'], dtype=float)
        inverse_t = np.array(saved['t_table_to_cam1'], dtype=float).reshape(3)
        if not np.allclose(rotation, inverse_r.T, atol=1e-6) or not np.allclose(translation, -rotation @ inverse_t, atol=1e-6):
            raise ValueError('Inconsistent table transform/inverse')
        if 'R_table_to_cam2' in saved:
            if not np.allclose(saved['R_table_to_cam2'], cal['R'] @ inverse_r, atol=1e-6):
                raise ValueError('Table frame and stereo rotation disagree; use matching calibration files')
        if 't_table_to_cam2' in saved:
            expected_t = cal['R'] @ inverse_t + cal['T'].reshape(3)
            if not np.allclose(np.asarray(saved['t_table_to_cam2']).reshape(3), expected_t, atol=1e-6):
                raise ValueError('Table frame and stereo translation disagree; use matching calibration files')
    return cal, size, rotation, translation


def image_landmarks(result, size):
    """Deliberately never read landmark z or hand_world_landmarks."""
    if len(result.hand_landmarks) != 1:
        return None, 'Unknown'
    landmarks = result.hand_landmarks[0]
    if len(landmarks) != 21:
        return None, 'Unknown'
    points = np.array([[point.x*size[0], point.y*size[1]] for point in landmarks], dtype=np.float64)
    label = 'Unknown'
    if result.handedness and result.handedness[0]:
        label = result.handedness[0][0].category_name
    return points, label


def reject(status, errors=None):
    return dict(status=status, points_cam1=None, points_table=None,
                errors=np.full(21, np.nan) if errors is None else errors,
                landmark_valid=np.zeros(21, dtype=bool),
                groups={name: False for name in GROUP_POINTS})


def usable_groups(points):
    """Check each control group's real measured points; never fill missing joints."""
    groups = {name: False for name in GROUP_POINTS}
    if points is None:
        return groups
    for name, required in GROUP_POINTS.items():
        if not np.isfinite(points[list(required)]).all():
            continue
        try:
            across = points[5]-points[17]
            a, b = points[5]-points[0], points[17]-points[0]
            if not all(0.02 < np.linalg.norm(v) < 0.22 for v in (across, a, b)):
                continue
            if np.linalg.norm(np.cross(g1.retarget.unit(a), g1.retarget.unit(b))) < 0.1:
                continue
            if name in ('wrist', 'thumb'):
                if not 0.02 < np.linalg.norm(points[9]-points[0]) < 0.22:
                    continue
                g1.palm_frame(*points[list(PALM)])
            if name != 'wrist':
                chain = (0, 1, 2, 3, 4) if name == 'thumb' else (0, *g1.retarget.CHAINS[name])
                lengths = np.linalg.norm(np.diff(points[list(chain)], axis=0), axis=1)
                if np.any(lengths < 0.0015) or np.any(lengths > 0.15):
                    continue
                if name != 'thumb':
                    finger_angles(points, name)
            groups[name] = True
        except (ValueError, FloatingPointError):
            continue
    return groups


def finger_angles(points, name):
    """Same flexion math as script 07, scoped to one independently valid chain."""
    mcp, pip, dip, tip = g1.retarget.CHAINS[name]
    normal = g1.retarget.unit(np.cross(points[5]-points[0], points[17]-points[0]))
    forward = g1.retarget.unit(points[mcp]-points[0])
    side = g1.retarget.unit(np.cross(normal, forward))
    proximal = g1.retarget.unit(points[pip]-points[mcp])
    projected = g1.retarget.unit(proximal-side*np.dot(proximal, side))
    names = g1.retarget.FINGERS[name]
    return {names[0]: 0.0,
            names[1]: float(np.arccos(np.clip(np.dot(forward, projected), -1, 1))),
            names[2]: g1.retarget.bend(points[mcp], points[pip], points[dip]),
            names[3]: g1.retarget.bend(points[pip], points[dip], points[tip])}


def reconstruct_hand(left, right, cal, table_r, table_t, max_reprojection=3.0):
    if left is None or right is None:
        return reject('MISSING_CAM1' if left is None else 'MISSING_CAM2')
    if left.shape != (21, 2) or right.shape != (21, 2):
        return reject('INVALID_2D')
    try:
        paired = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
        if not paired.any():
            return reject('INVALID_2D')
        # Triangulate separately so a singular correspondence cannot poison its neighbours.
        points = np.full((21, 3), np.nan)
        for index in np.flatnonzero(paired):
            try:
                with np.errstate(divide='raise', invalid='raise', over='raise'):
                    points[index] = stereo.triangulate(left[index:index+1], right[index:index+1], cal)[0]
            except (cv2.error, ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
        finite = np.isfinite(points).all(axis=1)
        if not finite.any():
            return reject('INVALID_TRIANGULATION')
        camera2 = points @ cal['R'].T + cal['T'].reshape(3)
        positive = finite & (points[:, 2] > 0) & (camera2[:, 2] > 0)
        errors = np.full(21, np.nan)
        indices = np.flatnonzero(positive)
        if not len(indices):
            return reject('BEHIND_CAMERA')
        rvec, _ = cv2.Rodrigues(cal['R'])
        with np.errstate(divide='raise', invalid='raise', over='raise'):
            projected1, _ = cv2.projectPoints(points[indices], np.zeros(3), np.zeros(3), cal['K1'], cal['D1'])
            projected2, _ = cv2.projectPoints(points[indices], rvec, cal['T'], cal['K2'], cal['D2'])
            errors[indices] = np.maximum(np.linalg.norm(projected1.reshape(-1, 2)-left[indices], axis=1),
                                        np.linalg.norm(projected2.reshape(-1, 2)-right[indices], axis=1))
        valid = positive & np.isfinite(errors) & (errors <= max_reprojection)
        points[~valid] = np.nan
        table = points @ table_r.T + table_t
        groups = usable_groups(table)
        if all(groups.values()) and valid.all():
            status = 'STEREO_VALID'
        elif any(groups.values()):
            status = 'PARTIAL_STEREO'
        else:
            status = 'REPROJECTION_REJECTED' if np.any(positive & ~valid) else 'HAND_GEOMETRY_REJECTED'
        return dict(status=status, points_cam1=points, points_table=table, errors=errors,
                    landmark_valid=valid, groups=groups)
    except (cv2.error, ValueError, FloatingPointError, np.linalg.LinAlgError):
        return reject('GEOMETRY_REJECTED')


class StereoWristTracker(g1.WristTracker):
    def update(self, points, now):
        moving = False
        if self.reference is None and points is not None and self.pending:
            basis = g1.palm_frame(*points[list(PALM)])
            window = [*self.pending, basis]
            moving = any(np.degrees(np.arccos(np.clip((np.trace(a.T @ b)-1)/2, -1, 1)))
                         > CALIBRATION_MAX_MOTION_DEG for i, a in enumerate(window) for b in window[i+1:])
            if moving:
                self.pending.clear()
        pose, status, error = super().update(points, now)
        return pose, 'CALIBRATION_MOVING' if moving else status, error


class StereoRetargeter:
    def __init__(self, model):
        self.model = model
        self.neutral = g1.setup_pose(model)
        self.bindings = g1.right_bindings(model)
        self.filters = {name: g1.live.LiveFilter(model, {joint: self.bindings[joint] for joint in joints})
                        for name, joints in g1.retarget.FINGERS.items()}
        self.pose = self.neutral.copy()
        self.thumb = g1.ThumbRetargeter(model, self.bindings)
        self.wrist = StereoWristTracker(model)

    def update(self, points_table, now):
        # Centre the measured points to make the no-translation contract explicit.
        relative = None if points_table is None else points_table-points_table[0]
        groups = usable_groups(relative)
        thumb_angles, thumb_state, thumb_error = self.thumb.update(relative if groups['thumb'] else None)
        states = {}
        for name, filter_ in self.filters.items():
            valid = groups[name] and (name != 'thumb' or thumb_state in ('TRACKING', 'LIMITED_FIT'))
            if valid:
                angles = thumb_angles if name == 'thumb' else finger_angles(relative, name)
                filtered, _ = filter_.update(angles, now)
                addresses = [self.bindings[joint][0] for joint in g1.retarget.FINGERS[name]]
                self.pose[addresses] = filtered[addresses]
                states[name] = 'TRACKING'
            else:
                # Freeze this group's applied pose, not its old target or a neutral substitute.
                filter_.history.clear()
                filter_.last_time = now
                states[name] = 'HOLD'
        wrist_pose, wrist_state, wrist_error = self.wrist.update(relative if groups['wrist'] else None, now)
        self.pose[self.wrist.addresses] = wrist_pose
        if self.wrist.reference is not None:
            calibration = 'Wrist calibration complete'
        elif not groups['wrist']:
            calibration = 'Wrist calibration waiting: reliable stereo palm needed (0/5)'
        elif wrist_state == 'CALIBRATION_MOVING':
            calibration = 'Wrist calibration restarted: hold still (1/5)'
        else:
            calibration = f'Wrist calibration started: hold neutral ({len(self.wrist.pending)}/5)'
        return self.pose.copy(), dict(finger_state=' | '.join(f'{name}:{state}' for name, state in states.items()),
                                      group_states=states, thumb_state=thumb_state, thumb_fit_mm=thumb_error,
                                      wrist_state=wrist_state, wrist_error_deg=wrist_error, calibration=calibration)


def draw_view(frame, points, tags, valid):
    image = frame.copy()
    if points is not None:
        finite = np.isfinite(points).all(axis=1)
        pixels = np.rint(np.where(np.isfinite(points), points, 0)).astype(int)
        for a, b in g1.live.CONNECTIONS:
            if finite[a] and finite[b]:
                cv2.line(image, tuple(pixels[a]), tuple(pixels[b]), (0, 210, 0) if valid[a] and valid[b] else (0, 0, 220), 2)
        for index, point in enumerate(pixels):
            if finite[index]:
                cv2.circle(image, tuple(point), 4, (0, 210, 0) if valid[index] else (0, 0, 220), -1)
    for tag_id, corners in tags.items():
        polygon = np.rint(corners).astype(np.int32)
        cv2.polylines(image, [polygon], True, (255, 190, 0), 2)
        cv2.putText(image, f'Tag {tag_id}', tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 190, 0), 2)
    image = cv2.resize(image, (640, 360))
    return image


def positive(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Value must be positive and finite')
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=int, default=2, help='Camera1 index from calibration')
    parser.add_argument('--right', type=int, default=0, help='Camera2 index from calibration')
    parser.add_argument('--left-video', type=Path)
    parser.add_argument('--right-video', type=Path)
    parser.add_argument('--calib', type=Path, default=CODE/'stereo/calibration/results_new/stereo_calibration.npz')
    parser.add_argument('--table-frame', type=Path, default=CODE/'stereo/calibration/results_new/table_frame_multi.npz')
    parser.add_argument('--hand-model', type=Path, default=ROOT/'models/mediapipe/hand_landmarker.task')
    parser.add_argument('--model', type=Path, default=ROOT/'models/scenario_g1_dex5_stationary.xml')
    parser.add_argument('--tag-ids', nargs='+', type=int, default=[4, 5])
    parser.add_argument('--tag-mm', type=positive, default=25.0)
    parser.add_argument('--max-reproj', type=positive, default=3.0)
    parser.add_argument('--max-grab-delta-ms', type=positive, default=50.0,
                        help='Reject live pairs with excessive host grab-time separation; not hardware synchronization')
    parser.add_argument('--flip-right-preview', action='store_true', help='Vertical DISPLAY flip only, matching the existing tracker')
    parser.add_argument('--view', choices=['full', 'right-hand'], default='right-hand')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/datetime.now().strftime('stereo_retarget_%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    if bool(args.left_video) != bool(args.right_video):
        parser.error('Provide BOTH --left-video and --right-video; a single video cannot supply stereo')
    if not args.left_video and args.left == args.right:
        parser.error('Stereo requires two different camera devices')
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    if args.headless and not args.left_video and not args.max_frames:
        parser.error('Headless camera tests require --max-frames')
    if len(set(args.tag_ids)) != len(args.tag_ids) or any(i < 0 for i in args.tag_ids):
        parser.error('Tag IDs must be distinct and nonnegative')
    return args


def quality_csv_header(controller):
    return ['frame', 'time_s', 'stereo_status', 'max_reprojection_px', 'host_grab_delta_ms',
            'label_cam1', 'label_cam2', 'shared_tags', 'valid_stereo_tags',
            'finger_state', 'thumb_state', 'thumb_fit_mm', 'wrist_state', 'wrist_error_deg',
            *g1.WRIST_NAMES, *controller.bindings,
            'valid_landmarks', 'landmark_valid_mask', 'wrist_calibration_frames',
            *[f'{name}_geometry_valid' for name in GROUP_POINTS],
            *['thumb_control_state' if name == 'thumb' else f'{name}_state' for name in controller.filters]]


def main():
    args = parse_args()
    cal, size, table_r, table_t = load_calibration(args.calib, args.table_frame)
    if not args.hand_model.is_file():
        raise FileNotFoundError(args.hand_model)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    controller = StereoRetargeter(model)
    data = mujoco.MjData(model)
    g1.retarget.apply_pose(model, data, controller.neutral)
    viewer, viewer_threads = None, []
    times, camera_points, table_points, pixels1, pixels2, valid, poses = [], [], [], [], [], [], []
    landmark_masks, group_masks = [], []
    counts, group_counts, update_counts = Counter(), Counter(), Counter()
    window = 'Stereo hand + wrist | stationary G1 Dex5'
    recorded_video = args.left_video is not None
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_detector = cv2.aruco.ArucoDetector(dictionary)
    try:
        with ExitStack() as stack:
            if recorded_video:
                cam1 = cv2.VideoCapture(str(args.left_video))
                stack.callback(cam1.release)
                cam2 = cv2.VideoCapture(str(args.right_video))
                stack.callback(cam2.release)
                if not cam1.isOpened() or not cam2.isOpened():
                    raise RuntimeError('Cannot open the paired stereo videos')
                fps = cam1.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps) or fps <= 0 or not np.isclose(fps, cam2.get(cv2.CAP_PROP_FPS), rtol=1e-4):
                    raise ValueError('Stereo videos must have matching valid FPS')
                if int(cam1.get(cv2.CAP_PROP_FRAME_COUNT)) != int(cam2.get(cv2.CAP_PROP_FRAME_COUNT)):
                    raise ValueError('Stereo videos must have matching frame counts and synchronized starts')
            else:
                cam1 = stereo.open_camera(args.left, size)
                stack.callback(cam1.release)
                cam2 = stereo.open_camera(args.right, size)
                stack.callback(cam2.release)
                cam1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cam2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fps = 30.0
            args.output.mkdir(parents=True, exist_ok=True)
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(args.hand_model)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
            hand1 = stack.enter_context(mp.tasks.vision.HandLandmarker.create_from_options(options))
            hand2 = stack.enter_context(mp.tasks.vision.HandLandmarker.create_from_options(options))
            if not args.headless:
                from mujoco import viewer as mj_viewer
                before = set(threading.enumerate())
                viewer = mj_viewer.launch_passive(model, data)
                viewer_threads = [t for t in threading.enumerate() if t not in before]
                viewer.cam.lookat[:] = data.site_xpos[controller.wrist.site] if args.view == 'right-hand' else [0, 0, 0.85]
                viewer.cam.distance = 0.65 if args.view == 'right-hand' else 2.4
                viewer.cam.azimuth, viewer.cam.elevation = 135, -15
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            print(f'STEREO ONLY: camera1={args.left}, camera2={args.right}, size={size}, tags={args.tag_ids}', flush=True)
            print(f'Calibration: {args.calib}\nTable frame: {args.table_frame}', flush=True)
            print('One physical RIGHT hand in both views. No monocular 3D fallback, arm translation, or walking.', flush=True)
            print('Hold neutral for 5 consecutive reliable palm frames (within 8 degrees). R recalibrates wrist; Q/Escape exits.', flush=True)
            print('Landmarks are gated separately. Invalid finger groups HOLD their last pose; red preview points are rejected.', flush=True)
            start, previous_ms, index = time.monotonic(), -1, 0
            log = stack.enter_context((args.output/'stereo_quality.csv').open('w', newline=''))
            writer = csv.writer(log)
            writer.writerow(quality_csv_header(controller))
            while viewer is None or viewer.is_running():
                begin = time.monotonic()
                ok1 = cam1.grab()
                grabbed1 = time.monotonic()
                ok2 = cam2.grab()
                grabbed2 = time.monotonic()
                if not ok1 or not ok2:
                    if recorded_video and not ok1 and not ok2:
                        break
                    raise RuntimeError('Stereo pair capture failed; stopping without monocular fallback')
                ok1, frame1 = cam1.retrieve()
                ok2, frame2 = cam2.retrieve()
                if not ok1 or not ok2:
                    raise RuntimeError('Stereo frame retrieval failed')
                if (frame1.shape[1], frame1.shape[0]) != size or (frame2.shape[1], frame2.shape[0]) != size:
                    raise ValueError(f'Both raw frames must match calibration size {size}; no silent resizing')
                now = index/fps if recorded_video else time.monotonic()-start
                ms = max(previous_ms+1, round(now*1000))
                previous_ms = ms
                rgb1, rgb2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                result1 = hand1.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb1)), ms)
                result2 = hand2.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb2)), ms)
                left, label1 = image_landmarks(result1, size)
                right, label2 = image_landmarks(result2, size)
                grab_delta = (grabbed2-grabbed1)*1000
                measurement = (reject('GRAB_TIMING_REJECTED') if not recorded_video and grab_delta > args.max_grab_delta_ms
                               else reconstruct_hand(left, right, cal, table_r, table_t, args.max_reproj))
                tags1 = {i: c for i, c in stereo.detect(frame1, tag_detector).items() if i in args.tag_ids}
                tags2 = {i: c for i, c in stereo.detect(frame2, tag_detector).items() if i in args.tag_ids}
                shared = sorted(set(tags1) & set(tags2))
                tag_valid = [i for i in shared if stereo.reconstruct(i, tags1[i], tags2[i], cal, args.tag_mm, args.max_reproj) is not None]
                pose, state = controller.update(measurement['points_table'], now)
                g1.retarget.apply_pose(model, data, pose)
                data.time = now
                if not np.isfinite(data.xpos).all():
                    raise RuntimeError('Nonfinite MuJoCo kinematics')
                accepted = measurement['status'] == 'STEREO_VALID'
                counts[measurement['status']] += 1
                group_counts.update(name for name, good in measurement['groups'].items() if good)
                update_counts.update(name for name, status in state['group_states'].items() if status == 'TRACKING')
                times.append(now)
                valid.append(accepted)
                landmark_masks.append(measurement['landmark_valid'])
                group_masks.append([measurement['groups'][name] for name in GROUP_POINTS])
                camera_points.append(measurement['points_cam1'] if measurement['points_cam1'] is not None else np.full((21, 3), np.nan))
                table_points.append(measurement['points_table'] if measurement['points_table'] is not None else np.full((21, 3), np.nan))
                pixels1.append(left if left is not None else np.full((21, 2), np.nan))
                pixels2.append(right if right is not None else np.full((21, 2), np.nan))
                poses.append(pose.copy())
                finite_errors = measurement['errors'][np.isfinite(measurement['errors'])]
                max_error = float(finite_errors.max()) if len(finite_errors) else None
                writer.writerow([index, now, measurement['status'], max_error, grab_delta, label1, label2,
                                 ','.join(map(str, shared)), ','.join(map(str, tag_valid)),
                                 state['finger_state'], state['thumb_state'], state['thumb_fit_mm'], state['wrist_state'], state['wrist_error_deg'],
                                 *pose[controller.wrist.addresses], *[pose[a] for a, _, _ in controller.bindings.values()],
                                 int(measurement['landmark_valid'].sum()),
                                 ''.join('1' if good else '0' for good in measurement['landmark_valid']),
                                 len(controller.wrist.pending),
                                 *[measurement['groups'][name] for name in GROUP_POINTS],
                                 *[state['group_states'][name] for name in controller.filters]])
                if viewer is not None:
                    viewer.sync()
                    image1 = draw_view(frame1, left, tags1, measurement['landmark_valid'])
                    image2 = draw_view(frame2, right, tags2, measurement['landmark_valid'])
                    if args.flip_right_preview:
                        image2 = cv2.flip(image2, 0)
                    view = np.hstack([image1, image2])
                    banner = np.zeros((155, 1280, 3), dtype=np.uint8)
                    lines = [f'{index}: {measurement["status"]} | cam1={args.left} {label1}, cam2={args.right} {label2}',
                             f'{state["calibration"]} | Wrist: {state["wrist_state"]}',
                             f'{state["finger_state"]} | Thumb fit: {state["thumb_state"]}',
                             f'Max reprojection: {max_error if max_error is not None else "n/a"} px | Tags {args.tag_ids}, shared {shared}, valid {tag_valid}',
                             f'Valid stereo points: {measurement["landmark_valid"].sum()}/21 (green); rejected: red | arm/body fixed | no monocular fallback']
                    for row, text in enumerate(lines):
                        cv2.putText(banner, text, (10, 25+row*29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
                    cv2.imshow(window, np.vstack([banner, view]))
                    key = cv2.waitKey(1) & 0xff
                    if key == ord('r'):
                        controller.wrist.reset()
                    if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if recorded_video:
                        time.sleep(max(0, 1/fps-(time.monotonic()-begin)))
                index += 1
                if args.max_frames is not None and index >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print('Stereo test interrupted.', flush=True)
    finally:
        if viewer is not None:
            viewer.close()
            for thread in viewer_threads:
                thread.join(timeout=5)
        if not args.headless:
            cv2.destroyAllWindows()
    poses = np.asarray(poses, dtype=float).reshape(-1, model.nq)
    controlled = set(controller.wrist.addresses.tolist()) | {a for a, _, _ in controller.bindings.values()}
    fixed = sorted(set(range(model.nq))-controlled)
    stationary = not len(poses) or bool(np.allclose(poses[:, fixed], controller.neutral[fixed], atol=1e-12))
    np.savez_compressed(args.output/'stereo_landmarks_and_targets.npz',
                        points_cam1_m=np.asarray(camera_points, dtype=np.float32).reshape(-1, 21, 3),
                        points_table_m=np.asarray(table_points, dtype=np.float32).reshape(-1, 21, 3),
                        pixels_cam1=np.asarray(pixels1, dtype=np.float32).reshape(-1, 21, 2),
                        pixels_cam2=np.asarray(pixels2, dtype=np.float32).reshape(-1, 21, 2),
                        timestamps=np.asarray(times), stereo_valid=np.asarray(valid, dtype=bool), qpos=poses,
                        landmark_valid=np.asarray(landmark_masks, dtype=bool).reshape(-1, 21),
                        group_valid=np.asarray(group_masks, dtype=bool).reshape(-1, len(GROUP_POINTS)),
                        group_names=np.array(list(GROUP_POINTS)),
                        source=np.array('stereo_triangulated_2d_landmarks'), monocular_fallback=np.array(False))
    report = dict(frames=len(times), statuses=dict(counts), accepted_stereo_frames=int(sum(valid)),
                  partial_stereo_frames=counts['PARTIAL_STEREO'],
                  group_geometry_valid_frames={name: group_counts[name] for name in GROUP_POINTS},
                  finger_update_frames={name: update_counts[name] for name in controller.filters},
                  wrist_calibrated=controller.wrist.reference is not None,
                  calibration_max_motion_deg=CALIBRATION_MAX_MOTION_DEG,
                  source='stereo_triangulated_2d_landmarks', monocular_fallback=False,
                  calibration=str(args.calib.resolve()), table_frame=str(args.table_frame.resolve()),
                  image_size=size, tag_ids=args.tag_ids, g1_body_and_arm_stationary=stationary,
                  robot_xyz_control=False, coordinates_mirrored=False,
                  limitations='One physical right hand. Per-landmark stereo and per-group geometry checks cannot prove anatomical accuracy. Invalid groups hold, not fresh measurements. Software grab timing is not hardware exposure synchronization. Existing relative wrist IK and thumb mapping limitations remain.',
                  privacy='Local inference; no camera images/video saved')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'Stereo frames={len(times)}, complete={sum(valid)}, partial={counts["PARTIAL_STEREO"]}, body/arm fixed={stationary}; output={args.output}', flush=True)


if __name__ == '__main__':
    main()
