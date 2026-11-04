"""Quality checks shared by the new Test 5 scripts (no robot control)."""
from pathlib import Path
import json
import numpy as np


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def load_tracking(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    n = len(data['detected'])
    if not n or data['landmarks'].shape != (n, 21, 3):
        raise ValueError('Expected nonempty landmarks with shape (frames, 21, 3)')
    for key in ('timestamps', 'handedness'):
        if data[key].shape != (n,):
            raise ValueError(f'{key} is not frame-aligned')
    if data['world_landmarks'].shape != (n, 21, 3):
        raise ValueError('world_landmarks is not frame-aligned')
    if not np.isfinite(data['timestamps']).all() or np.any(np.diff(data['timestamps']) <= 0):
        raise ValueError('Timestamps must be finite and strictly increasing')
    fps = float(data['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive and finite')
    if not np.isfinite(data['landmarks'][data['detected'].astype(bool)]).all():
        raise ValueError('Detected landmarks contain nonfinite values')
    return data


def summarize(data):
    detected = data['detected'].astype(bool)
    t, fps = data['timestamps'], float(data['fps'])
    indices = np.flatnonzero(detected)
    labels, counts = np.unique(data['handedness'][detected], return_counts=True)
    gaps = []
    for a, b in runs(~detected):
        gaps.append(dict(start_frame=a, end_frame=b-1, frames=b-a,
                         start_s=float(t[a]), duration_s=float(t[b-1]-t[a]+1/fps),
                         internal=bool(a > 0 and b < len(detected))))
    flips = [int(i) for i in range(1, len(detected))
             if detected[i-1] and detected[i] and data['handedness'][i-1] != data['handedness'][i]]
    # Only consecutive observations: never measure a jump across a missing interval.
    jumps = []
    for i in range(1, len(detected)):
        if detected[i-1] and detected[i]:
            previous = data['landmarks'][i-1]
            scale = max(float(np.linalg.norm(previous[0, :2]-previous[9, :2])), 1e-6)
            displacement = float(np.median(np.linalg.norm(data['landmarks'][i, :, :2]-previous[:, :2], axis=1))/scale)
            if displacement > 0.75:
                jumps.append(dict(frame=i, displacement_palm_lengths=displacement))
    active_rate = None
    if len(indices):
        active_rate = float(detected[indices[0]:indices[-1]+1].mean()*100)
    return dict(frames=len(detected), detected=int(detected.sum()), missing=int((~detected).sum()),
                detection_percent=float(detected.mean()*100), fps=fps,
                first_detected_frame=int(indices[0]) if len(indices) else None,
                last_detected_frame=int(indices[-1]) if len(indices) else None,
                detected_span_percent=active_rate,
                detected_span_note='First-to-last detection span is NOT a hand-visibility ground truth annotation.',
                missing_runs=gaps, raw_handedness_counts=dict(zip(labels.tolist(), counts.tolist())),
                consecutive_label_flip_frames=flips, possible_jump_frames=jumps)


def fill_short_gaps(landmarks, world, detected, timestamps, max_gap_s=0.15):
    """Interpolate bounded short gaps only; preserve raw observations separately."""
    if not np.isfinite(max_gap_s) or max_gap_s < 0:
        raise ValueError('max_gap_s must be finite and nonnegative')
    result, result_world = landmarks.copy(), world.copy()
    interpolated = np.zeros(len(detected), dtype=bool)
    dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0
    for a, b in runs(~np.asarray(detected, dtype=bool)):
        if a == 0 or b == len(detected) or timestamps[b-1]-timestamps[a]+dt > max_gap_s + 1e-9:
            continue
        for values in (result, result_world):
            if np.isfinite(values[[a-1, b]]).all():
                alpha = ((timestamps[a:b]-timestamps[a-1])/(timestamps[b]-timestamps[a-1]))[:, None, None]
                values[a:b] = (1-alpha)*values[a-1] + alpha*values[b]
        interpolated[a:b] = np.isfinite(result[a:b]).all(axis=(1, 2))
    return result, result_world, interpolated


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
