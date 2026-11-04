"""Run the existing SAM3 cup detector and known-size fitter on an annotated take."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from test_sam3_cup import create_predictor, segment_images
from estimate_cup_xyz import estimate_cup

ACTUAL_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTUAL_CODE))
from stereo.reconstruct_approach import annotation_interval, verify_source, read_rows
from trajectory_io import sha256
from g1_workspace import load_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--take', type=Path, required=True)
    parser.add_argument('--annotation', type=Path, default=ACTUAL_CODE/'annotation1.json')
    parser.add_argument('--calib', type=Path, default=ACTUAL_CODE/'stereo/calibration/parallel_results_v3/stereo_calibration.npz')
    parser.add_argument('--table', type=Path, default=ACTUAL_CODE/'stereo/calibration/parallel_results_v3/table_frame_multi.npz')
    parser.add_argument('--workspace', type=Path, default=ACTUAL_CODE.parent/'config/g1_right_workspace.json')
    parser.add_argument('--model', type=Path, default=Path('/home/sparx3d/physikal_workflow/sam3.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--frames', type=int, nargs='+')
    args = parser.parse_args()
    start, end, video = annotation_interval(args.annotation)
    timestamps = read_rows(args.take/'timestamps.csv')
    source = verify_source(args.take, video, start, end, len(timestamps))
    frames = args.frames or np.linspace(start, end-1, 3).astype(int).tolist()
    if any(frame < start or frame >= end for frame in frames):
        raise ValueError('All requested frames must lie in the annotated Approach')
    rotation, origin, profile = load_workspace(args.workspace)
    if sha256(args.table) != profile['source_table_calibration_sha256']:
        raise ValueError('Workspace and source v3 table calibration disagree')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(4, args.threads))
    print(f'Loading existing SAM3 model on CPU; frames {frames}; no extra Camera 2 rotation.', flush=True)
    predictor = create_predictor(args.model, 'cpu')
    captures = [cv2.VideoCapture(str(args.take/f'camera{index}.avi')) for index in (1,2)]
    results = []
    try:
        for frame in frames:
            directory = args.output/f'frame_{frame:06d}'
            directory.mkdir()
            for index, capture in enumerate(captures, 1):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f'Could not decode Camera {index} frame {frame}')
                cv2.imwrite(str(directory/f'camera{index}.png'), image)
            masks = segment_images(directory, predictor)
            result = dict(frame=frame, masks=masks)
            if all(masks[name]['found'] for name in ('camera1', 'camera2')):
                fit = estimate_cup(directory, args.calib, args.table)
                result.update({key: value.tolist() if isinstance(value, np.ndarray) else value for key,value in fit.items()})
                result['accepted'] = (fit['camera1_dice'] >= .55 and fit['camera2_dice'] >= .55
                                      and fit['camera_disagreement_mm'] <= 30)
            else:
                result.update(accepted=False, reason='Missing cup mask in a view')
            results.append(result)
            (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            print(f'Frame {frame}: accepted={result["accepted"]}', flush=True)
    finally:
        for capture in captures:
            capture.release()
    good = [result for result in results if result['accepted']]
    report = dict(source_take=str(args.take.resolve()), source_verification=source,
                  approach_frames=[start,end], sampled_frames=frames, results=results,
                  reused_detector='test_sam3_cup.py SAM3SemanticPredictor, prompt cup, largest mask',
                  reused_fit='estimate_cup_xyz.py upright 82 mm diameter x 98 mm known cylinder, two-view silhouette fit',
                  table_calibration_sha256=sha256(args.table), workspace_sha256=sha256(args.workspace),
                  orientation_estimated=False, base_height_constrained_to_table=True)
    if len(good) >= 2:
        positions = np.array([result['base_xyz_table_m'] for result in good])
        base = np.median(positions, axis=0)
        spread = float(np.max(np.linalg.norm(positions-base, axis=1)))
        world = rotation @ base+origin
        report.update(accepted=spread <= .02, accepted_frames=len(good), base_xyz_table_m=base.tolist(),
                      base_xyz_world_m=world.tolist(), maximum_temporal_deviation_m=spread,
                      placement_note='Position only. Upright Z is assumed from the tabletop; handle yaw is not measured.')
    else:
        report.update(accepted=False, reason='Fewer than two consistent accepted stereo fits; do not move simulation cup')
    (args.output/'cup_pose_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('results','source_verification')}, indent=2), flush=True)
    return 0 if report['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
