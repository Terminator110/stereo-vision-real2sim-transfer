"""Sample right wrist + waist-yaw IK and optionally regenerate the pink MuJoCo sites."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from g1_hand_ik import HandIK
from g1_workspace import CollisionChecker, export_world


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/datetime.now().strftime('waist_wrist_workspace_%Y%m%d_%H%M%S'))
    parser.add_argument('--update-visual', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    model = mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial = model.qpos0.copy()
    arm = HandIK(model, 'right', include_waist_yaw=True)
    checker = CollisionChecker(model, include_waist_yaw=True)
    wrist = int(arm.ids[-2])
    waist = int(arm.ids[-1])
    shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'g1_01_right_shoulder_pitch_link')
    shoulder_initial = data.xpos[shoulder].copy()
    waist_initial = data.xanchor[waist].copy()
    shoulder_circle_radius = float(np.linalg.norm((shoulder_initial-waist_initial)[:2]))
    radius = 0.
    body = int(model.jnt_bodyid[wrist])
    while body != shoulder:
        radius += float(np.linalg.norm(model.body_pos[body]))
        body = int(model.body_parentid[body])
    profile_path = ROOT/'config/g1_right_workspace.json'
    profile = json.loads(profile_path.read_text())
    assert profile['yaw_degrees'] == -90
    origin = np.array(profile['origin_world_m'])
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'work_table_top')
    near = float(data.geom_xpos[table, 0]-model.geom_size[table, 0])
    depths, reds, heights = np.linspace(0, .5, 11), np.linspace(-.3, .3, 13), np.linspace(0, .25, 6)
    old = ROOT/'outputs/right_wrist_reachable_volume_20261103'
    with (old/'samples.csv').open() as handle:
        old_rows = [r for r in csv.DictReader(handle) if r['status']=='REACHABLE_STATIC']
    with np.load(old/'reachable_qpos.npz', allow_pickle=False) as cache:
        old_poses = cache['qpos'].copy()
    if len(old_rows) != len(old_poses):
        raise ValueError('Baseline pose/sample indexing mismatch')
    baseline = {tuple(round(float(row[k]), 7) for k in ('x', 'y', 'z')): pose
                for row, pose in zip(old_rows, old_poses)}
    seeds = [initial[arm.addresses].copy()]
    for angle in (-1.2, -.6, .6, 1.2):
        seed = seeds[0].copy()
        seed[-1] = angle
        seed[3] = .8
        seeds.append(seed)
    rng = np.random.default_rng(20261103)
    seeds += [rng.uniform(arm.lower*.75, arm.upper*.75) for _ in range(3)]
    warm = None

    def point_at(qpos):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        return data.xanchor[wrist].copy()

    def solve(target, extra=None, relative_left=False):
        nonlocal warm
        # With waist yaw, the shoulder moves on a horizontal circle. The old
        # fixed-shoulder spherical rejection would wrongly exclude new reaches.
        if not relative_left:
            radial = np.linalg.norm((target-waist_initial)[:2])
            lower_bound = np.hypot(radial-shoulder_circle_radius, target[2]-shoulder_initial[2])
            if lower_bound > radius+.002:
                return dict(status='OUTSIDE_CONSERVATIVE_YAW_REACH_BOUND', error_m=None, collisions=[]), None
        key = tuple(round(float(v), 7) for v in target)
        cached = baseline.get(key) if not relative_left else None
        if cached is not None and np.linalg.norm(point_at(cached)-target) <= .002:
            hits = checker.check(cached)
            if not hits:
                warm = cached[arm.addresses].copy()
                return dict(status='REACHABLE', error_m=float(np.linalg.norm(point_at(cached)-target)),
                            collisions=[], preserved_arm_only_solution=True), cached.copy()
        best, closest_hits, converged_any = float('inf'), [], False

        def residual(q):
            data.qpos[:] = initial
            data.qpos[arm.addresses] = q
            mujoco.mj_kinematics(model, data)
            desired = data.xanchor[left_wrist]+target if relative_left else target
            return 10*(data.xanchor[wrist]-desired)

        starts = ([extra] if extra is not None else [])+([warm] if warm is not None else [])+seeds
        for seed in starts:
            fit = least_squares(residual, seed, bounds=(arm.lower, arm.upper), max_nfev=100,
                                ftol=1e-7, xtol=1e-7, gtol=1e-7)
            error = float(np.linalg.norm(residual(fit.x))/10)
            best = min(best, error)
            if error > .002:
                continue
            converged_any = True
            pose = initial.copy()
            pose[arm.addresses] = fit.x
            hits = checker.check(pose)
            if hits:
                closest_hits = hits[:3]
                continue
            warm = fit.x.copy()
            return dict(status='REACHABLE', error_m=error, collisions=[],
                        preserved_arm_only_solution=False), pose
        return dict(status='COLLISION_REJECTED' if converged_any else 'NO_IK_FOUND',
                    error_m=best, collisions=closest_hits), None

    rows, poses, success_xyz = [], [], []
    grid = np.zeros((6, 11, 13), bool)
    for di, depth in enumerate(depths):
        for ri, red in enumerate(reds):
            for hi, height in enumerate(heights):
                target = np.array([near+depth, origin[1]-red, origin[2]+height])
                result, pose = solve(target)
                row = dict(depth_m=float(depth), red_m=float(red), height_m=float(height),
                           x=float(target[0]), y=float(target[1]), z=float(target[2]), **result)
                if pose is not None:
                    grid[hi, di, ri] = True
                    fixed = np.setdiff1d(np.arange(model.nq), arm.addresses)
                    assert np.array_equal(pose[fixed], initial[fixed])
                    assert np.all(pose[arm.addresses] >= arm.lower) and np.all(pose[arm.addresses] <= arm.upper)
                    row['waist_yaw_deg'] = float(np.degrees(pose[arm.addresses[-1]]))
                    row['qpos_index'] = len(poses)
                    poses.append(pose)
                    success_xyz.append(target)
                rows.append(row)
        print(f'Depth {depth*100:.0f} cm: {int(grid[:,di,:].sum())}/78 accepted', flush=True)
        # Keep partial evidence if a long CPU-only recomputation is interrupted.
        write_json(out/'progress.json', dict(depth_slices_completed=di+1, accepted=int(grid.sum()), samples=len(rows)))

    mujoco.mj_forward(model, data)
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    left_wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'g1_01_left_wrist_yaw_joint')
    left_hand_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'g1_01_dex5_left_frame')
    finger_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'g1_01_dex5_left_Link_24L')
    left_points = {'initial_left_wrist_anchor':data.xanchor[left_wrist].copy(),
                   'initial_left_hand_base':data.site_xpos[left_hand_site].copy(),
                   'initial_left_index_distal_link_origin':data.xpos[finger_body].copy()}
    checks = []
    for label, point in left_points.items():
        for offset in (0, .05):
            target = point+np.array([0,0,offset])
            result, pose = solve(target)
            checks.append(dict(label=label+('_5cm_above' if offset else ''), target_world_m=target.tolist(),
                               left_reference='Initial world position; left hand moves with torso yaw', **result,
                               waist_yaw_deg=float(np.degrees(pose[arm.addresses[-1]])) if pose is not None else None))
    # This deliberately asks both wrist anchors to coincide. It is a contact/
    # overlap test, not a claim that one hand cannot approach the other.
    result, pose = solve(np.zeros(3), relative_left=True)
    checks.append(dict(label='coincident_current_left_right_wrist_anchors',
                       left_reference='Follows the left wrist as torso rotates', **result))
    write_json(out/'left_hand_area_checks.json', checks)

    cells = np.ones((5,10,12), bool)
    for h in (0,1):
        for d in (0,1):
            for r in (0,1):
                cells &= grid[h:h+5,d:d+10,r:r+12]
    summary = dict(target='Right wrist-yaw joint anchor', orientation='Unconstrained',
                   controlled_joints=arm.joint_names, other_joint_angles_fixed=True,
                   shoulder_bound='Shoulder-circle lower distance bound accounting for waist yaw, not a fixed-shoulder sphere',
                   total_samples=len(rows), accepted_samples=len(poses),
                   preserved_arm_only_samples=sum(row.get('preserved_arm_only_solution',False) for row in rows),
                   previous_arm_only_samples=len(old_rows), spacing_m=.05,
                   desired_box_m=[.60,.50,.25], height_counts=[int(grid[i].sum()) for i in range(6)],
                   heights_m=heights.tolist(), at_20cm_accepted=int(grid[4].sum()),
                   grid_cell_volume_estimate_litres=float(cells.sum()*.125),
                   volume_estimate_note='5 cm cells with all eight corners accepted; interiors and connecting motions unverified',
                   collision_policy=checker.description(), left_hand_checks='left_hand_area_checks.json',
                   initial_root_pose_changed=False, scene_placement_changed=False,
                   elapsed_s=time.monotonic()-start,
                   limitations='Finite multistart local IK; failure is not proof of global unreachability. Static states, not motion planning.')
    write_json(out/'report.json', summary)
    write_json(out/'samples.json', rows)
    np.savez_compressed(out/'reachable_qpos.npz', qpos=np.asarray(poses), xyz=np.asarray(success_xyz))
    if args.update_visual:
        visual = ROOT/'models/scenes/g1_right_wrist_reachable_volume.xml'
        shutil.copy2(visual, out/'previous_visual.xml')
        xml = ET.Element('mujoco', model='right_wrist_waist_yaw_reachable_samples')
        xml.append(ET.Comment('Pink = right wrist + waist yaw static position IK; full upper-body collision checks. Recompute after scene changes.'))
        body = ET.SubElement(ET.SubElement(xml,'worldbody'),'body',name='g1_right_wrist_reachable_samples')
        for i, point in enumerate(success_xyz):
            ET.SubElement(body,'site',name=f'right_wrist_reachable_{i:04d}',type='sphere',
                          pos=' '.join(f'{v:.9f}' for v in point), size='0.006',rgba='1 0.08 0.75 0.75')
        ET.indent(xml,space='  ')
        ET.ElementTree(xml).write(visual,encoding='unicode')
        profile['right_wrist_volume_report'] = str(out/'report.json')
        write_json(profile_path,profile)
        world = ROOT/'outputs/approach_shared_workspace_20261102/approach_mujoco.csv'
        export_world(world.with_name('approach_smoothed_table.csv'),profile_path,world)
    print(json.dumps(summary,indent=2),flush=True)
    print('LEFT HAND CHECKS:',json.dumps(checks,indent=2),flush=True)


if __name__ == '__main__':
    main()
