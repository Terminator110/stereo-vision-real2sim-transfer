"""Reconstruct one annotated Approach from ONE AprilTag; never invent alignment."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

import cv2
import numpy as np
from scipy.signal import savgol_filter

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from trajectory_io import proper_rotation, scene_table_frame, sha256, write_xyz
from g1_workspace import export_world, load_workspace


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def annotation_interval(path):
    intervals = []
    for task in json.loads(Path(path).read_text()):
        for annotation in task.get('annotations', []):
            for result in annotation.get('result', []):
                value = result.get('value', {})
                if 'Approach' in value.get('timelinelabels', []):
                    for interval in value.get('ranges', []):
                        start, end = interval['start'], interval['end']
                        if start != int(start) or end != int(end) or not 0 <= start < end:
                            raise ValueError('Approach requires nonnegative integer frame bounds')
                        intervals.append((int(start), int(end), task['data']['video']))
    if len(intervals) != 1:
        raise ValueError('Require exactly one annotated Approach interval')
    return intervals[0]


def verify_source(take, annotation_video, start, end, timestamp_count, uploaded=None):
    relative = Path(urlparse(annotation_video).path)
    parts = relative.parts
    if parts[:3] != ('/', 'data', 'upload') or '..' in parts:
        raise ValueError('Supply a Label Studio upload reference, not an inferred recording')
    uploaded = Path(uploaded) if uploaded else Path.home()/'.local/share/label-studio/media'/Path(*parts[2:])
    video = take/'label_studio.mp4'
    if not uploaded.is_file() or sha256(uploaded) != sha256(video):
        raise ValueError('Annotated uploaded video does not hash-match this take')
    captures = [cv2.VideoCapture(str(take/name)) for name in ('label_studio.mp4', 'camera1.avi', 'camera2.avi')]
    errors = []
    try:
        counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in captures]
        if not all(c.isOpened() for c in captures) or len(set(counts)) != 1 or counts[0] != timestamp_count or end > counts[0]:
            raise ValueError('Video pair, label video, annotation and acquisition frame counts disagree')
        for frame in sorted({0, start, (start+end)//2, end-1, counts[0]-1}):
            images = []
            for capture in captures:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if not ok:
                    raise ValueError('Could not decode source verification frame')
                images.append(image)
            label, cam1, cam2 = images
            h, w = label.shape[:2]
            if w % 2:
                raise ValueError('Expected side-by-side stereo label video')
            pair_errors = [float(np.mean(np.abs(label[:, i*w//2:(i+1)*w//2].astype(float)
                                                - cv2.resize(camera, (w//2, h)).astype(float))))
                           for i, camera in enumerate((cam1, cam2))]
            if max(pair_errors) > 8:
                raise ValueError('Label video frames do not match this camera pair after resizing')
            errors.append(dict(frame=frame, camera_pair_mae=pair_errors))
    finally:
        for capture in captures:
            capture.release()
    return dict(upload=str(uploaded), sha256=sha256(video), frames=counts[0],
                sampled_video_matches=errors, note='Encoded FPS is NOT used for motion timing')


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(('min', 'median', 'p95', 'max'), map(float, np.quantile(values, [0, .5, .95, 1])))) if len(values) else None


def isolated_outliers(times, points, valid, max_speed, deviation):
    rejected = np.zeros(len(valid), dtype=bool)
    ids = np.flatnonzero(valid)
    for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
        vin = (points[b]-points[a])/(times[b]-times[a])
        vout = (points[c]-points[b])/(times[c]-times[b])
        speeds = np.array([np.linalg.norm(vin), np.linalg.norm(vout)])
        predicted = points[a]+(points[c]-points[a])*(times[b]-times[a])/(times[c]-times[a])
        reversal = np.dot(vin, vout)/(np.prod(speeds)+1e-12)
        neighbour_speed = np.linalg.norm(points[c]-points[a])/(times[c]-times[a])
        # Flag an isolated spike, not a sustained fast but locally consistent motion.
        rejected[b] = (np.all(speeds > max_speed) and reversal < -.8
                       and neighbour_speed < max_speed and np.linalg.norm(points[b]-predicted) > deviation)
    return rejected


def reconstruct_tag(frames, times, rows, args):
    points = np.full((len(frames), 3), np.nan)
    errors = np.full(len(frames), np.nan)
    valid_flag = np.zeros(len(frames), dtype=bool)
    for i, frame in enumerate(frames):
        row = rows.get(int(frame))
        if row is None:
            continue
        for axis, key in enumerate(('x_table_m', 'y_table_m', 'z_table_m')):
            points[i, axis] = float(row[key]) if row.get(key) else np.nan
        errors[i] = float(row['reproj_px']) if row.get('reproj_px') else np.nan
        valid_flag[i] = int(row['valid']) == 1
    measured = valid_flag & np.isfinite(points).all(axis=1)
    reliable = measured & np.isfinite(errors) & (errors <= args.max_reproj)
    spikes = isolated_outliers(times, points, reliable, args.max_speed, args.outlier_deviation)
    reliable &= ~spikes
    filtered = np.full_like(points, np.nan)
    filtered[reliable] = points[reliable]
    interpolated = np.zeros(len(frames), dtype=bool)
    gaps = []
    missing = np.flatnonzero(~reliable)
    runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1)+1) if len(missing) else []
    for run in runs:
        a, b = int(run[0]-1), int(run[-1]+1)
        bounded = a >= 0 and b < len(frames)
        duration = float(times[b]-times[a]) if bounded else float(times[run[-1]]-times[run[0]])
        fill = bounded and len(run) <= args.max_missing_frames and duration <= args.max_gap_seconds
        policy = ('INTERPOLATED' if fill else 'BOUNDARY_NO_EXTRAPOLATION' if not bounded
                  else 'REVIEW_NOT_FILLED' if len(run) <= 10 else 'LONG_NOT_FILLED')
        gaps.append(dict(start_frame=int(frames[run[0]]), end_frame=int(frames[run[-1]]),
                         missing_frames=len(run), bounding_elapsed_s=duration, policy=policy))
        if fill:
            for axis in range(3):
                filtered[run, axis] = np.interp(times[run], times[[a, b]], points[[a, b], axis])
            interpolated[run] = True
    usable = np.flatnonzero(np.isfinite(filtered).all(axis=1))
    cuts = [i for i in range(1, len(usable)) if usable[i]-usable[i-1] > 1
            or times[usable[i]]-times[usable[i-1]] > args.max_gap_seconds]
    segments = []
    for ids in np.split(usable, cuts) if len(usable) else []:
        local_t = times[ids]
        original = filtered[ids].copy()
        smooth = original.copy()
        smoothing = 'SKIPPED_TOO_SHORT'
        duration = float(local_t[-1]-local_t[0])
        if args.smoothing_window >= 3 and duration > 0:
            grid = np.linspace(local_t[0], local_t[-1], max(2, round(duration*args.resample_hz)+1))
            if len(grid) >= args.smoothing_window:
                regular = np.column_stack([np.interp(grid, local_t, original[:, j]) for j in range(3)])
                regular = savgol_filter(regular, args.smoothing_window, 2, axis=0, mode='interp')
                candidate = np.column_stack([np.interp(local_t, grid, regular[:, j]) for j in range(3)])
                if np.max(np.linalg.norm(candidate-original, axis=1)) <= args.max_smoothing_displacement:
                    smooth, smoothing = candidate, 'LIGHT_SAVGOL_TIME_UNIFORM_GRID'
                else:
                    smoothing = 'SKIPPED_EXCESSIVE_DISPLACEMENT'
        filtered[ids] = smooth
        n = int(reliable[ids].sum())
        fraction = n/len(ids)
        length = float(np.linalg.norm(np.diff(smooth, axis=0), axis=1).sum())
        meaningful = n >= 5 and duration >= .5 and fraction >= .5 and length >= .01
        grid = local_t[0]+np.arange(int(np.floor(duration*args.resample_hz+1e-9))+1)/args.resample_hz
        # Retain the exact physical endpoint; only the terminal interval may be <50 ms.
        if local_t[-1]-grid[-1] > 1e-9:
            grid = np.r_[grid, local_t[-1]]
        export = np.column_stack([np.interp(grid, local_t, smooth[:, j]) for j in range(3)])
        segments.append(dict(ids=ids, times=grid-local_t[0], points=export,
                             start_frame=int(frames[ids[0]]), end_frame=int(frames[ids[-1]]),
                             duration_s=duration, original_samples=n, filled_frames=int(interpolated[ids].sum()),
                             measured_fraction=fraction, path_length_m=length, meaningful=meaningful,
                             smoothing=smoothing, max_smoothing_displacement_m=float(np.max(np.linalg.norm(smooth-original, axis=1)))))
    ids = np.flatnonzero(reliable)
    velocities = np.diff(points[ids], axis=0)/np.diff(times[ids])[:, None] if len(ids) > 1 else np.empty((0, 3))
    accelerations = np.diff(velocities, axis=0)/((times[ids[2:]]-times[ids[:-2]])/2)[:, None] if len(ids) > 2 else np.empty((0, 3))
    meaningful = [s for s in segments if s['meaningful']]
    longest = max(meaningful, key=lambda s: s['duration_s'], default=None)
    stats = dict(valid_measurements=int(measured.sum()), reliable_measurements=int(reliable.sum()),
                 coverage=float(reliable.mean()), reprojection_px=quantiles(errors[measured & np.isfinite(errors)]),
                 rejected_spatial_frames=frames[spikes].tolist(), rejected_quality_frames=frames[measured & ~reliable & ~spikes].tolist(),
                 longest_missing_frames=max((g['missing_frames'] for g in gaps), default=0), missing_intervals=gaps,
                 velocity_m_s=quantiles(np.linalg.norm(velocities, axis=1)),
                 acceleration_estimate_m_s2=quantiles(np.linalg.norm(accelerations, axis=1)),
                 velocity_note='Differences between actual measured timestamps; across gaps these are averages, not recovered motion',
                 longest_meaningful_duration_s=longest['duration_s'] if longest else 0.0,
                 interpolated_frames=int(interpolated.sum()), segments=[{k: v for k, v in s.items() if k not in ('ids', 'times', 'points')} for s in segments])
    return dict(raw=points, filtered=filtered, errors=errors, measured=measured, reliable=reliable,
                spikes=spikes, interpolated=interpolated, gaps=gaps, segments=segments, longest=longest, stats=stats)


def select_tag(results):
    eligible = [tag for tag, value in results.items() if value['longest'] is not None]
    if not eligible:
        return None, 'No tag has a meaningful supported segment (>=5 original samples, >=0.5 s, >=50% measured, >=1 cm travel)'
    best = max(eligible, key=lambda tag: (results[tag]['stats']['longest_meaningful_duration_s'],
                                        results[tag]['stats']['coverage'],
                                        -results[tag]['stats']['reprojection_px']['p95']))
    if 5 in eligible and results[5]['stats']['coverage'] >= .8*results[best]['stats']['coverage'] and results[5]['longest']['duration_s'] >= .9*results[best]['longest']['duration_s']:
        return 5, 'Tag 5 has comparable continuity/coverage and is anatomically preferred'
    return best, 'Longest meaningful single-tag segment; coverage tie-break. No tag-centre mixing or wrist offset'


def save_plot(path, frames, times, results, selected, transform=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(221, projection='3d')
    for tag, result in results.items():
        p = result['raw'][result['measured']]
        if len(p):
            ax.scatter(*p.T, label=f'Tag {tag} raw', s=13)
    if selected is not None:
        result = results[selected]
        for segment in result['segments']:
            ax.plot(*segment['points'].T, color='black', linewidth=2 if segment is result['longest'] else 1)
        p = result['raw'][result['measured']]
        extent = max(.15, float(np.abs(p[:, :2]).max())+.05)
    else:
        extent = .4
    x, y = np.meshgrid(np.linspace(-extent, extent, 2), np.linspace(-extent, extent, 2))
    ax.plot_surface(x, y, np.zeros_like(x), alpha=.15, color='grey')
    ax.scatter(0, 0, 0, color='red', label='v3 tag-reference origin (NOT table centre)')
    ax.set(xlabel='v3 X (m)', ylabel='v3 Y (m)', zlabel='v3 Z (m)', title='Measured/calibrated tabletop plane; patch is NOT physical table edges')
    ax.legend(fontsize=7)
    ax = fig.add_subplot(222)
    if selected is not None:
        result = results[selected]
        for j, axis in enumerate('xyz'):
            ax.scatter(times, result['raw'][:, j], s=10, label=f'raw {axis}')
            for segment in result['segments']:
                ids = segment['ids']
                ax.plot(times[ids], result['filtered'][ids, j], linewidth=1)
        for gap in result['gaps']:
            if gap['policy'] != 'INTERPOLATED':
                a = int(gap['start_frame']-frames[0]); b = int(gap['end_frame']-frames[0])
                ax.axvspan(times[a], times[b], color='red', alpha=.12)
    ax.set(xlabel='Actual seconds from Approach start', ylabel='Metres', title='Unfilled gaps shaded; never joined by filtering')
    ax.legend(fontsize=8)
    ax = fig.add_subplot(223, projection='3d')
    if transform is not None and selected is not None:
        rotation, origin, _ = transform
        p = results[selected]['longest']['points'] @ rotation.T+origin
        ax.plot(*p.T, color='blue')
        x, y = np.meshgrid([.39, 1.01], [-.375, .375])
        ax.plot_surface(x, y, np.full_like(x, .762), alpha=.2, color='grey')
        ax.scatter(*origin, color='red')
        ax.set(xlabel='MuJoCo world X (m)', ylabel='MuJoCo world Y (m)', zlabel='World Z up (m)', title='Shared workspace placement, scale 1 (NOT absolute physical alignment)')
    else:
        ax.text2D(.05, .5, 'Reconstruction ready in the v3 frame.\nSelect a collision-checked shared G1 workspace\nwith select_g1_workspace.py, then export/replay.\nNo physical table-centre measurement needed.', transform=ax.transAxes)
    ax = fig.add_subplot(224)
    ax.axis('off')
    lines = [f'Tag {tag}: {r["stats"]["reliable_measurements"]}/{len(frames)} reliable, longest missing {r["stats"]["longest_missing_frames"]} frames' for tag, r in results.items()]
    if selected is not None:
        s = results[selected]['longest']
        lines += [f'Selected: Tag {selected}, frames {s["start_frame"]}-{s["end_frame"]}',
                  f'{s["original_samples"]} measured + {s["filled_frames"]} short-gap frames',
                  f'Longest supported segment: {s["duration_s"]:.3f} s',
                  'Not a reconstruction of the full Approach unless all frames are supported.']
    ax.text(.02, .9, '\n'.join(lines), va='top')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--take', type=Path, required=True)
    ap.add_argument('--annotation', type=Path, default=CODE/'annotation1.json')
    ap.add_argument('--uploaded-video', type=Path)
    ap.add_argument('--table-calibration', type=Path, default=CODE/'stereo/calibration/parallel_results_v3/table_frame_multi.npz')
    ap.add_argument('--workspace', type=Path, help='Reusable shared manipulation workspace profile; no physical table alignment')
    ap.add_argument('--model', type=Path, default=CODE.parent/'models/scenario_g1_dex5_stationary.xml')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--max-reproj', type=float, default=3.0)
    ap.add_argument('--max-speed', type=float, default=2.0)
    ap.add_argument('--outlier-deviation', type=float, default=.04)
    ap.add_argument('--max-missing-frames', type=int, default=3, choices=range(4))
    ap.add_argument('--max-gap-seconds', type=float, default=.5)
    ap.add_argument('--smoothing-window', type=int, default=5)
    ap.add_argument('--max-smoothing-displacement', type=float, default=.01)
    ap.add_argument('--resample-hz', type=float, default=20.0)
    args = ap.parse_args()
    for name in ('max_reproj', 'max_speed', 'outlier_deviation', 'max_gap_seconds', 'max_smoothing_displacement', 'resample_hz'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            ap.error(f'{name} must be positive/finite')
    if args.smoothing_window != 0 and (args.smoothing_window < 3 or args.smoothing_window % 2 == 0):
        ap.error('Smoothing window must be zero (off) or an odd number >=3')
    take = args.take.resolve()
    start, end, video = annotation_interval(args.annotation)
    stamps = read_rows(take/'timestamps.csv')
    frame_list = [int(row['frame']) for row in stamps]
    all_times = np.array([float(row['time_s']) for row in stamps])
    if frame_list != list(range(len(stamps))) or not np.isfinite(all_times).all() or np.any(np.diff(all_times) <= 0):
        raise ValueError('Acquisition timestamps must have unique contiguous frame indices and increasing physical times')
    provenance = verify_source(take, video, start, end, len(stamps), args.uploaded_video)
    frames = np.arange(start, end)
    times = all_times[frames]-all_times[start]
    rows = read_rows(take/'tags_xyz_table.csv')
    original = {(int(r['frame']), int(r['tag_id'])): r for r in read_rows(take/'tags_xyz.csv')}
    with np.load(args.table_calibration, allow_pickle=False) as cal:
        rotation = proper_rotation(cal['R_cam1_to_table'])
        translation = np.asarray(cal['t_cam1_to_table']).reshape(3)
        camera_above = bool(translation[2] > 0 and np.dot(rotation[2], -np.asarray(cal['table_origin_cam1_m']).reshape(3)) > 0)
        if not camera_above:
            raise ValueError('Saved table normal does not point upward toward the above-table Camera 1')
        calibration = dict(plane_rms_mm=float(cal['plane_rms_mm']), plane_p95_mm=float(cal['plane_p95_mm']),
                           camera1_height_table_m=float(translation[2]), positive_z_toward_above_table_camera=True,
                           origin_is_first_tag3_sample_not_physical_table_centre=True)
    seen = set()
    maps = {tag: {} for tag in (2, 4, 5)}
    for row in rows:
        frame, tag = int(row['frame']), int(row['tag_id'])
        if tag not in maps or not start <= frame < end:
            continue
        key = (frame, tag)
        if key in seen or key not in original or abs(float(row['time_s'])-all_times[frame]) > 1e-6:
            raise ValueError('Duplicated/mismatched tracking frame identity or timing')
        seen.add(key)
        if int(row['valid']) != int(original[key]['valid']):
            raise ValueError('Original and table tracking validity disagree')
        if int(row['valid']):
            p = np.array([float(original[key][k]) for k in ('x_cam1_m', 'y_cam1_m', 'z_cam1_m')])
            q = np.array([float(row[k]) for k in ('x_table_m', 'y_table_m', 'z_table_m')])
            if not np.allclose(q, rotation @ p+translation, atol=1e-6):
                raise ValueError('Table measurements do not match the supplied v3 calibration')
        maps[tag][frame] = row
    results = {tag: reconstruct_tag(frames, times, maps[tag], args) for tag in maps}
    selected, rationale = select_tag(results)
    transform = load_workspace(args.workspace) if args.workspace else None
    output = args.output or take/datetime.now().strftime('approach_reconstruction_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    import mujoco
    scene_r, scene_centre = scene_table_frame(mujoco.MjModel.from_xml_path(str(args.model.resolve())))
    report = dict(approach_frames=[start, end], end_exclusive=True, frame_count=len(frames),
                  approach_last_frame_elapsed_s=float(times[-1]), selected_tag_id=selected, selection_rationale=rationale,
                  source_verification=provenance, source_take=str(take), calibration=calibration,
                  candidate_tags={str(k): v['stats'] for k, v in results.items()},
                  full_approach_reconstructed=False, workspace_profile_selected=transform is not None,
                  mapping_blocker=None if transform else 'Run select_g1_workspace.py to place the shared v3 origin in a reachable simulation workspace',
                  table_centred_to_world_rotation=scene_r.tolist(), tabletop_centre_world_m=scene_centre.tolist(),
                  hand_base_offset_m=[0, 0, 0], orientation='Preserve initial robot hand orientation; none measured',
                  thresholds={name: getattr(args, name) for name in ('max_reproj', 'max_speed', 'outlier_deviation', 'max_missing_frames', 'max_gap_seconds', 'smoothing_window', 'max_smoothing_displacement', 'resample_hz')})
    if selected is not None:
        result = results[selected]
        chosen = result['longest']
        report['selected_segment'] = {k: v for k, v in chosen.items() if k not in ('ids', 'times', 'points')}
        report['full_approach_reconstructed'] = bool(chosen['start_frame'] == start and chosen['end_frame'] == end-1)
        write_xyz(output/'approach_raw_table.csv', times[result['measured']], result['raw'][result['measured']])
        write_xyz(output/'approach_smoothed_table.csv', chosen['times'], chosen['points'])
        source_metadata = dict(coordinate_frame='calibrated_table_v3', units='metres',
            csv_sha256=sha256(output/'approach_smoothed_table.csv'), selected_tag_id=selected,
            table_calibration_sha256=sha256(args.table_calibration),
            hand_base_offset_m=[0, 0, 0], raw_table_csv=str((output/'approach_raw_table.csv').resolve()),
            raw_table_csv_sha256=sha256(output/'approach_raw_table.csv'), source_take=str(take),
            selected_segment=report['selected_segment'], resample_hz=args.resample_hz,
            all_approach_reconstructed=report['full_approach_reconstructed'])
        (output/'approach_smoothed_table.json').write_text(json.dumps(source_metadata, indent=2)+'\n')
        with (output/'approach_diagnostics.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['frame', 'time_s', 'tag_id', 'x_raw', 'y_raw', 'z_raw', 'x_filtered', 'y_filtered', 'z_filtered', 'reproj_px', 'status', 'interpolated', 'reason'])
            for i, frame in enumerate(frames):
                rejected = result['measured'][i] and not result['reliable'][i]
                status = ('REJECTED_OUTLIER' if rejected else 'ORIGINAL_VALID' if result['reliable'][i]
                          else 'INTERPOLATED' if result['interpolated'][i] else 'MISSING')
                reason = 'isolated_velocity_reversal' if result['spikes'][i] else 'reprojection_or_invalid_quality' if rejected else ''
                writer.writerow([frame, times[i], selected, *result['raw'][i], *result['filtered'][i], result['errors'][i], status, int(result['interpolated'][i]), reason])
        directory = output/'segments'
        directory.mkdir()
        for i, segment in enumerate(result['segments'], 1):
            write_xyz(directory/f'segment_{i:03d}_table.csv', segment['times'], segment['points'])
        if transform is not None:
            export_world(output/'approach_smoothed_table.csv', args.workspace, output/'approach_mujoco.csv')
    if transform is None:
        (output/'workspace_REQUIRED.txt').write_text('Select a shared workspace using code/select_g1_workspace.py --trajectory approach_smoothed_table.csv. No physical table-centre coordinates are required.\n')
    save_plot(output/'approach_trajectory_preview.png', frames, times, results, selected, transform)
    (output/'approach_reconstruction_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['Approach reconstruction (single tag, physical acquisition timestamps)',
             f'Frames {start}-{end-1}; selected tag: {selected}; {rationale}',
             f'Shared workspace selected: {transform is not None}; {report["mapping_blocker"] or "shared workspace mapping"}',
             f'MuJoCo tabletop centre WORLD: {scene_centre.tolist()} (not world origin)',
             'The v3 origin maps to the selected simulation workspace origin; +Z maps to world up. Horizontal alignment is adjustable yaw only. Scale stays 1.',
             'Tag-to-hand-base offset: ZERO by user request, not an anatomical calibration.',
             'Velocity/acceleration between observations do not recover motion hidden in gaps.',
             'Savgol smoothing is computed on a per-segment uniform temporary grid, then mapped back to real timestamps. Export is 20 Hz plus the exact final endpoint when its last interval is shorter than 50 ms.',
             'Raw CSV times are relative to Approach start; smoothed/segment exports restart at zero. No extrapolation or cross-tag averaging.',
             json.dumps(report, indent=2, allow_nan=False)]
    (output/'approach_reconstruction_report.txt').write_text('\n'.join(lines)+'\n')
    print(f'Output: {output}\nSelected tag: {selected}\nShared workspace selected: {transform is not None}')
    if selected is not None:
        print(f'Supported segment: frames {chosen["start_frame"]}-{chosen["end_frame"]}, {chosen["duration_s"]:.3f}s; NOT necessarily the full Approach')
    return 0 if selected is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
