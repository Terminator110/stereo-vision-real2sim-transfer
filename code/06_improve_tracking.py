"""Separate tracking experiment: VIDEO pass, IMAGE retries, bounded gap filling."""
import argparse
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from tracking_quality import fill_short_gaps, load_tracking, save_json, summarize

ROOT = Path(__file__).resolve().parents[1]
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
               (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def confidence(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError('Confidence must be in [0, 1]')
    return value


def extract(result):
    if not result.hand_landmarks:
        return None
    xyz = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
    if xyz.shape != (21, 3) or not np.isfinite(xyz).all():
        return None
    world = np.full((21, 3), np.nan, dtype=np.float32)
    if result.hand_world_landmarks:
        world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]], dtype=np.float32)
    label, score = 'Unknown', float('nan')
    if result.handedness and result.handedness[0]:
        category = result.handedness[0][0]
        label, score = category.category_name, float(category.score)
    return xyz, world, label, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=ROOT/'videos/vid_1.mp4')
    parser.add_argument('--model', type=Path, default=ROOT/'models/mediapipe/hand_landmarker.task')
    parser.add_argument('--baseline', type=Path, default=ROOT/'outputs/hand_landmarks.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/datetime.now().strftime('test5_improved_%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--detection-confidence', type=confidence, default=0.5)
    parser.add_argument('--presence-confidence', type=confidence, default=0.5)
    parser.add_argument('--tracking-confidence', type=confidence, default=0.5)
    parser.add_argument('--retry-confidence', type=confidence, default=0.35)
    parser.add_argument('--max-gap-s', type=float, default=0.15)
    parser.add_argument('--physical-hand', choices=['Right', 'Left'], default='Right')
    args = parser.parse_args()
    if not np.isfinite(args.max_gap_s) or args.max_gap_s < 0:
        parser.error('--max-gap-s must be finite and nonnegative')
    baseline = load_tracking(args.baseline)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0 or not np.isclose(fps, float(baseline['fps']), rtol=1e-4):
        cap.release()
        raise ValueError('Invalid FPS or baseline/video FPS mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    xyzs, worlds, labels, scores, sources, times = [], [], [], [], [], []
    video_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=args.detection_confidence,
        min_hand_presence_confidence=args.presence_confidence,
        min_tracking_confidence=args.tracking_confidence)
    image_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=args.retry_confidence,
        min_hand_presence_confidence=args.retry_confidence)
    primary_detected, previous_ms = [], -1
    try:
        with ExitStack() as stack:
            video_tracker = stack.enter_context(mp.tasks.vision.HandLandmarker.create_from_options(video_options))
            image_detector = stack.enter_context(mp.tasks.vision.HandLandmarker.create_from_options(image_options))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                i = len(times)
                ms = max(previous_ms+1, round(i*1000/fps))
                previous_ms = ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                value = extract(video_tracker.detect_for_video(image, ms))
                primary_detected.append(value is not None)
                source = 'video' if value else 'missing'
                # Independent palm detection can recover a failed tracking frame.
                # Lower-threshold candidates are explicitly marked, not called ground truth.
                if value is None:
                    value = extract(image_detector.detect(image))
                    if value:
                        source = 'image_retry'
                if value is None:
                    value = (np.full((21,3), np.nan, dtype=np.float32),
                             np.full((21,3), np.nan, dtype=np.float32), 'Unknown', float('nan'))
                xyz, world, label, score = value
                xyzs.append(xyz); worlds.append(world); labels.append(label); scores.append(score)
                sources.append(source); times.append(ms/1000)
                if (i+1) % 60 == 0:
                    print(f'Processed {i+1} frames', flush=True)
    finally:
        cap.release()
    if len(times) != len(baseline['detected']):
        raise ValueError('Baseline and input video frame counts differ')
    detected = np.array(sources) != 'missing'
    xyzs, worlds, times = np.asarray(xyzs), np.asarray(worlds), np.asarray(times)
    processed, processed_world, interpolated = fill_short_gaps(xyzs, worlds, detected, times, args.max_gap_s)
    # Keep raw MediaPipe classification independent of the operator's physical-hand assumption.
    np.savez_compressed(args.output/'hand_landmarks.npz', landmarks=xyzs, world_landmarks=worlds,
                        timestamps=times, detected=detected, handedness=np.array(labels), fps=fps,
                        handedness_score=np.array(scores), source=np.array(sources),
                        assumed_physical_hand=np.array(args.physical_hand),
                        processed_landmarks=processed, processed_world_landmarks=processed_world,
                        interpolated=interpolated, valid=detected|interpolated,
                        primary_detected=np.array(primary_detected))
    report = summarize(dict(landmarks=xyzs, world_landmarks=worlds, timestamps=times,
                            detected=detected, handedness=np.array(labels), fps=fps))
    report.update(baseline=summarize(baseline), video=str(args.video.resolve()), model=str(args.model.resolve()),
                  settings={key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()},
                  primary_detections=int(sum(primary_detected)), image_retry_candidates=int(sources.count('image_retry')),
                  recovered_baseline_missing_frames=np.flatnonzero(detected & ~baseline['detected']).tolist(),
                  lost_baseline_detected_frames=np.flatnonzero(~detected & baseline['detected']).tolist(),
                  interpolated_frames=np.flatnonzero(interpolated).tolist(),
                  warning='Retries and interpolation require visual review; detection count is not accuracy. No mirroring or robot control.')
    report['world_valid_frames'] = int(np.isfinite(processed_world).all(axis=(1, 2)).sum())
    save_json(args.output/'report.json', report)
    render_review(args.video, args.output/'review.mp4', fps, processed, detected, interpolated, sources, labels)
    print(f'Observed {detected.sum()}/{len(detected)}; retries={sources.count("image_retry")}; interpolated={interpolated.sum()}')
    print(f'Results: {args.output}')


def render_review(video_path, output, fps, points, detected, interpolated, sources, labels):
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        for i in range(len(points)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Video could not be decoded during review rendering')
            h, w = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot write {output}')
            color = (0,200,0) if sources[i] == 'video' else (0,170,255)
            if detected[i] or interpolated[i]:
                pixels = np.rint(points[i,:,:2]*[w,h]).astype(int)
                for a,b in CONNECTIONS:
                    cv2.line(frame, tuple(pixels[a]), tuple(pixels[b]), color, 2)
                for point in pixels:
                    cv2.circle(frame, tuple(point), 3, color, -1)
            status = 'INTERPOLATED - NOT OBSERVED' if interpolated[i] else sources[i]
            cv2.putText(frame, f'{i}: {status}; raw label={labels[i]}', (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == '__main__':
    main()
