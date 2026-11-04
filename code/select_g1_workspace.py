"""Select a reusable tabletop manipulation reference using IK and collisions."""
import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from g1_hand_ik import HandIK, HandTarget, quaternion
from g1_workspace import CollisionChecker, export_world
from trajectory_io import load_xyz, scene_table_frame, sha256


def margin(arm, qpos):
    q = qpos[arm.addresses]
    return float(np.minimum(q-arm.lower, arm.upper-q).min())


def validate_path(arm, checker, points, orientation, initial, swept=True):
    previous = initial.copy()
    minimum = float('inf')
    poses = []
    for index, point in enumerate(points):
        result = arm.solve(previous, HandTarget(point, orientation))
        if not result.converged:
            return dict(ok=False, sample=index, reason='IK_NOT_CONVERGED', error_m=result.position_error_m), []
        hits = checker.transition(previous, result.candidate_qpos, arm.addresses) if swept else checker.check(result.candidate_qpos)
        if hits:
            return dict(ok=False, sample=index, reason='COLLISION', collisions=hits[:3]), []
        previous = result.candidate_qpos
        minimum = min(minimum, margin(arm, previous))
        poses.append(previous.copy())
    return dict(ok=True, min_joint_margin_rad=minimum, samples=len(points)), poses


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trajectory', type=Path, required=True, help='Reconstructed single-tag v3 CSV')
    ap.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1]/'models/scenario_g1_dex5_stationary.xml')
    ap.add_argument('--output', type=Path, required=True, help='New reusable workspace JSON; never overwritten')
    ap.add_argument('--yaw-deg', type=float, help='Explicit v3-to-world Z rotation; default tests a bounded angle set')
    ap.add_argument('--export-dir', type=Path, help='Also export approach_mujoco.csv and metadata here')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Workspace already exists; reuse it or choose a new profile path')
    if args.yaw_deg is not None and not np.isfinite(args.yaw_deg):
        ap.error('Yaw must be finite')
    trajectory = load_xyz(args.trajectory)
    source = json.loads(args.trajectory.with_suffix('.json').read_text())
    if source.get('coordinate_frame') != 'calibrated_table_v3' or source.get('csv_sha256') != sha256(args.trajectory):
        ap.error('Trajectory must have matching reconstructed v3 provenance metadata')
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    arm, checker = HandIK(model, 'right'), CollisionChecker(model)
    initial = model.qpos0.copy()
    _, rotation = arm.hand_pose(initial)
    orientation = quaternion(rotation)
    _, centre = scene_table_frame(model)
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'work_table_top')
    if not np.allclose(model.geom_size[table, :2], [.31, .375]):
        ap.error('Workspace scan currently requires the existing world-aligned table')
    front = float(centre[0]-.31)
    records, candidates = [], []
    for x in (front, front+.015):
        for y in (-.12, -.18, -.06):
            origin = np.array([x, y, centre[2]])
            reachable, high_layer = 0, 0
            centre_margin = 0.0
            for height in (.18, .22, .26):
                for dx in (-.02, 0., .02):
                    for dy in (-.025, 0., .025):
                        p = origin+[dx, dy, height]
                        result = arm.solve(initial, HandTarget(p, orientation))
                        hits = checker.check(result.candidate_qpos) if result.converged else []
                        valid = result.converged and not hits
                        reachable += int(valid)
                        high_layer += int(valid and height == .26)
                        if valid and dx == 0 and dy == 0 and height == .26:
                            centre_margin = margin(arm, result.candidate_qpos)
                        records.append(dict(origin_x=x, origin_y=y, x=p[0], y=p[1], z=p[2],
                                            reachable=valid, ik_error_m=result.position_error_m,
                                            collision=bool(hits), joint_margin_rad=margin(arm, result.candidate_qpos) if valid else 0.0))
            candidates.append(dict(origin=origin, reachable=reachable, high_layer=high_layer, centre_margin=centre_margin))
            print(f'Anchor {origin.round(3).tolist()}: {reachable}/27 safe IK probes, upper working layer {high_layer}/9', flush=True)
    candidates.sort(key=lambda c: (c['high_layer'], c['reachable'], c['centre_margin']), reverse=True)
    valid_origins = [c for c in candidates if c['high_layer'] >= 6 and c['centre_margin'] >= .04]
    if not valid_origins:
        raise ValueError('No practical tabletop anchor found with useful two-axis movement and joint margin; table/model change requires operator approval')
    angles = [args.yaw_deg] if args.yaw_deg is not None else [-30., -60., -90., 0., 30., 60., 90., 180.]
    fits = []
    sample_ids = np.unique(np.linspace(0, len(trajectory.times)-1, 5).astype(int))
    for anchor in valid_origins:
        for yaw in angles:
            r = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
            points = trajectory.points @ r.T+anchor['origin']
            check, _ = validate_path(arm, checker, points[sample_ids], orientation, initial, swept=False)
            if check['ok']:
                fits.append((anchor, yaw, points, check['min_joint_margin_rad']))
        if fits:
            break
    fits.sort(key=lambda f: f[3], reverse=True)
    selected = None
    failures = []
    for anchor, yaw, points, _ in fits:
        result, poses = validate_path(arm, checker, points, orientation, initial)
        if result['ok']:
            selected = (anchor, yaw, result, poses)
            break
        failures.append(dict(yaw_degrees=yaw, **result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scan = args.output.with_name(args.output.stem+'_reachability.csv')
    with scan.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if selected is None:
        args.output.with_name(args.output.stem+'_FAILED.json').write_text(json.dumps(dict(reason='No unscaled, collision-checked placement found', failures=failures), indent=2)+'\n')
        raise ValueError('No safe mapping found; no profile created and no arm moved')
    anchor, yaw, check, poses = selected
    profile = dict(mapping_mode='shared_manipulation_workspace', source_frame='calibrated_table_v3',
                   origin_world_m=anchor['origin'].tolist(), yaw_degrees=yaw, scale=1,
                   source_origin_maps_directly=True, physical_table_alignment_required=False,
                   source_normal=[0, 0, 1], target_normal=[0, 0, 1], hand_base_offset_m=[0, 0, 0],
                   model=str(args.model.resolve()), model_root_sha256=sha256(args.model),
                   source_table_calibration_sha256=source.get('table_calibration_sha256'),
                   operating_probe_offsets_m=dict(x=[-.02, .02], y=[-.025, .025], height=[.18, .26]),
                   origin_note='Coordinate reference on tabletop; hand-base commands operate ABOVE it. This is not a command to press the palm into the tabletop.',
                   safe_probes=anchor['reachable'], total_probes=27, upper_layer_safe=anchor['high_layer'],
                   centre_joint_margin_rad=anchor['centre_margin'], collision_checks=checker.description(),
                   selected_path_check=check, chosen_from_current_approach=True,
                   reuse_policy='Fixed transform for future v3 recordings; do not re-fit or recenter each new demonstration. Revalidate every new path.',
                   selection_trajectory_sha256=sha256(args.trajectory))
    args.output.write_text(json.dumps(profile, indent=2)+'\n')
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        export_world(args.trajectory, args.output, args.export_dir/'approach_mujoco.csv')
    print(f'Workspace: {args.output}; origin={profile["origin_world_m"]}, yaw={yaw} degrees. No scale/timing changes.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
