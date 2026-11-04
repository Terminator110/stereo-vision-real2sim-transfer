"""Standalone G1 hand-position/orientation IK demonstration. No stereo or walking."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import threading
import time
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from g1_hand_ik import ARM_JOINTS, HandIKController, HandTarget, quaternion

ROOT = Path(__file__).resolve().parents[1]


def demo_targets(controller, starts, elapsed, duration):
    phase = elapsed/duration
    fraction = min(1.0, phase*3) if phase <= 2/3 else max(0.0, (1-phase)*3)
    fraction = fraction*fraction*(3-2*fraction)
    targets = {}
    for side in controller.arms:
        sign = 1 if side == 'left' else -1
        position, rotation = starts[side]
        p = position+fraction*np.array([0.10, sign*0.02, 0.06])
        r = Rotation.from_rotvec([0, 0, sign*0.18*fraction]).as_matrix() @ rotation
        targets[side] = HandTarget(p, quaternion(r))
    return targets


def add_goal_markers(viewer, targets):
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        for side, target in targets.items():
            if scene.ngeom >= scene.maxgeom:
                break
            color = [0.2, 0.9, 0.4, 0.85] if side == 'left' else [1.0, 0.55, 0.1, 0.85]
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.full(3, 0.012), target.position, np.eye(3).ravel(), np.array(color))
            scene.ngeom += 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'models/scenario_g1_dex5_stationary.xml')
    parser.add_argument('--side', choices=['left', 'right', 'both'])
    parser.add_argument('--waist-yaw', action='store_true',
                        help='Include torso left/right rotation with right-arm IK; waist roll/pitch remain fixed')
    parser.add_argument('--trajectory', type=Path, help='Reconstructed approach_mujoco.csv with shared-workspace metadata; marker-only by default')
    parser.add_argument('--execute-trajectory', action='store_true')
    parser.add_argument('--start-at-trajectory', action='store_true',
                        help='Simulation only: initialize at the first collision-checked IK pose before opening the viewer; skip preposition motion')
    parser.add_argument('--workspace-reviewed', action='store_true', help='Confirm marker-only workspace preview was reviewed before arm execution')
    parser.add_argument('--speed', type=float, default=1.0, help='Recorded trajectory playback speed, (0,1]; .5 is half speed')
    parser.add_argument('--max-trajectory-gap', type=float, default=.15)
    parser.add_argument('--preposition-seconds', type=float, default=6.0)
    parser.add_argument('--target', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Absolute MuJoCo-world metres, single arm only; default is the demo trajectory')
    parser.add_argument('--quat', type=float, nargs=4, metavar=('W', 'X', 'Y', 'Z'),
                        help='Target hand-site orientation; default preserves its starting orientation')
    parser.add_argument('--duration', type=float, default=12.0)
    parser.add_argument('--hz', type=int, default=50)
    parser.add_argument('--max-velocity', type=float, default=1.0, help='Maximum arm joint speed, rad/s')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--hide-reachability', action='store_true',
                        help='Hide only the pink reachable-wrist spheres; keep origin, cameras and workspace outline visible')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/datetime.now().strftime('g1_hand_ik_%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    if args.trajectory:
        if args.side not in (None, 'right') or args.target is not None or args.quat is not None:
            parser.error('CSV replay is right-arm XYZ only; do not combine --target, --quat or another side')
        args.side = 'right'
    else:
        args.side = args.side or 'both'
        if args.execute_trajectory or args.workspace_reviewed:
            parser.error('Trajectory execution/review flags require --trajectory')
    if args.start_at_trajectory and not (args.trajectory and args.execute_trajectory):
        parser.error('--start-at-trajectory requires --trajectory and --execute-trajectory')
    if args.waist_yaw and args.side != 'right':
        parser.error('--waist-yaw requires --side right or --trajectory (shared waist is not solved independently for both arms)')
    if not np.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error('--speed must be in (0,1]')
    if not np.isfinite(args.max_trajectory_gap) or args.max_trajectory_gap <= 0 or not np.isfinite(args.preposition_seconds) or args.preposition_seconds <= 0:
        parser.error('Gap threshold and preposition timeout must be positive/finite')
    if not np.isfinite(args.duration) or args.duration <= 0 or not 10 <= args.hz <= 120:
        parser.error('Use a positive finite duration and 10-120 Hz')
    if args.target is not None and args.side == 'both':
        parser.error('--target requires --side left or --side right')
    if args.quat is not None and args.target is None:
        parser.error('--quat requires --target')
    if not np.isfinite(args.max_velocity) or args.max_velocity <= 0:
        parser.error('--max-velocity must be positive and finite')
    return args


def main():
    args = parse_args()
    if args.trajectory:
        from trajectory_replay import replay
        try:
            return replay(args)
        except (ValueError, FileNotFoundError, KeyError) as error:
            print(f'Trajectory replay blocked before further motion: {error}', file=sys.stderr)
            return 2
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if args.hide_reachability:
        cloud = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'g1_right_wrist_reachable_samples')
        if cloud >= 0:
            model.site_rgba[model.site_bodyid == cloud, 3] = 0
    sides = ('left', 'right') if args.side == 'both' else (args.side,)
    controller = HandIKController(model, sides, args.max_velocity, include_waist_yaw=args.waist_yaw)
    checker, collision_failure = None, None
    if args.waist_yaw:
        from g1_workspace import CollisionChecker
        checker = CollisionChecker(model, include_waist_yaw=True)
    starts = {side: arm.hand_pose(controller.qpos) for side, arm in controller.arms.items()}
    fixed_targets = None
    if args.target is not None:
        side = sides[0]
        fixed_targets = {side: HandTarget(np.array(args.target), quaternion(starts[side][1])
                                         if args.quat is None else np.array(args.quat))}
        p, r = controller.arms[side].validate_target(controller.qpos, fixed_targets[side])
        fixed_targets[side] = HandTarget(p, quaternion(r))
    data = mujoco.MjData(model)
    data.qpos[:] = controller.qpos
    mujoco.mj_forward(model, data)
    args.output.mkdir(parents=True, exist_ok=False)
    viewer, viewer_threads = None, []
    times, poses, target_positions, target_quaternions = [], [], [], []
    records = {side: [] for side in sides}
    step_s = 1/args.hz
    steps = max(1, int(np.ceil(args.duration*args.hz)))
    wall_start = time.monotonic()
    print(f'G1 hand IK: {", ".join(sides)}; world metres; hand-site quaternions w,x,y,z.', flush=True)
    print('Kinematic qpos + mj_forward only. No stereo, finger commands, walking, torque, collision avoidance, or hardware.', flush=True)
    print(f'Arm joint speed limit: {args.max_velocity} rad/s. Outputs: {args.output}', flush=True)
    try:
        if not args.headless:
            from mujoco import viewer as mj_viewer
            before = set(threading.enumerate())
            viewer = mj_viewer.launch_passive(model, data)
            viewer_threads = [t for t in threading.enumerate() if t not in before]
            viewer.cam.lookat[:] = [0.3, 0, 0.95]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 135, -18
            wall_start = time.monotonic()
        with (args.output/'ik_trace.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            fields = ['status', 'reached', 'solver_converged', 'position_error_m', 'orientation_error_deg',
                      'solver_position_error_m', 'solver_orientation_error_deg', 'solver_evaluations']
            writer.writerow(['frame', 'time_s', 'side', *fields, 'target_x', 'target_y', 'target_z',
                             'target_qw', 'target_qx', 'target_qy', 'target_qz', *ARM_JOINTS,
                             *(['waist_yaw'] if args.waist_yaw else [])])
            next_render = 0.0
            previous_time = 0.0
            for index in range(1, steps+1):
                if viewer is not None and not viewer.is_running():
                    break
                elapsed = min(index*step_s, args.duration)
                targets = fixed_targets or demo_targets(controller, starts, elapsed, args.duration)
                before_pose = controller.qpos.copy()
                pose, states = controller.step(targets, elapsed-previous_time)
                if checker is not None:
                    hits = checker.transition(before_pose, pose, controller.arms['right'].addresses)
                    if hits:
                        controller.qpos[:] = before_pose
                        collision_failure = dict(status='COLLISION_HOLD', collisions=hits[:3])
                        print(f'Waist-yaw IK stopped before colliding motion: {collision_failure}', flush=True)
                        break
                previous_time = elapsed
                data.qpos[:] = pose
                data.time = elapsed
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.xpos).all():
                    raise RuntimeError('Nonfinite MuJoCo kinematics')
                times.append(elapsed)
                poses.append(pose)
                target_positions.append([targets[side].position for side in sides])
                target_quaternions.append([targets[side].quaternion for side in sides])
                for side, state in states.items():
                    records[side].append(state)
                    writer.writerow([index, elapsed, side, *[state[name] for name in fields],
                                     *targets[side].position, *targets[side].quaternion,
                                     *pose[controller.arms[side].addresses]])
                if viewer is not None:
                    wall_elapsed = time.monotonic()-wall_start
                    if wall_elapsed >= next_render:
                        add_goal_markers(viewer, targets)
                        viewer.sync()
                        next_render = wall_elapsed+1/30
                    time.sleep(max(0.0, wall_start+elapsed-time.monotonic()))
    except KeyboardInterrupt:
        print('IK test interrupted.', flush=True)
    finally:
        if viewer is not None:
            viewer.close()
            for thread in viewer_threads:
                thread.join(timeout=5)
    poses = np.asarray(poses).reshape(-1, model.nq)
    addresses = np.concatenate([arm.addresses for arm in controller.arms.values()])
    fixed = np.setdiff1d(np.arange(model.nq), addresses)
    unchanged = bool(not len(poses) or np.array_equal(poses[:, fixed], np.broadcast_to(controller.initial_qpos[fixed], (len(poses), len(fixed)))))
    bounds_ok = all(bool(np.all(poses[:, arm.addresses] >= arm.lower-1e-10)
                         and np.all(poses[:, arm.addresses] <= arm.upper+1e-10)) for arm in controller.arms.values())
    delta = np.diff(np.vstack([controller.initial_qpos, poses]), axis=0)
    intervals = np.diff(np.r_[0.0, times])
    max_speed = float(np.max(np.abs(delta[:, addresses])/intervals[:, None])) if len(poses) else 0.0
    speed_ok = max_speed <= args.max_velocity+1e-9
    summaries = {}
    for side, rows in records.items():
        summaries[side] = dict(final=rows[-1] if rows else None,
                               reached_frames=sum(row['reached'] for row in rows),
                               solver_failed_frames=sum(not row['solver_converged'] for row in rows),
                               max_position_error_m=max((row['position_error_m'] for row in rows), default=None),
                               max_orientation_error_deg=max((row['orientation_error_deg'] for row in rows), default=None))
    complete = bool(len(times) == steps)
    success = complete and collision_failure is None and unchanged and bounds_ok and speed_ok and all(rows and rows[-1]['reached'] for rows in records.values())
    report = dict(success=bool(success), completed=complete, frames=len(times), sides=summaries,
                  waist_yaw_enabled=args.waist_yaw, controlled_joints={side:arm.joint_names for side,arm in controller.arms.items()},
                  collision_failure=collision_failure, collision_checks=checker.description() if checker else None,
                  simulated_duration_s=times[-1] if times else 0.0, wall_duration_s=time.monotonic()-wall_start,
                  other_joints_unchanged=unchanged, joint_limits_ok=bounds_ok,
                  max_joint_speed_rad_s=max_speed, configured_speed_rad_s=args.max_velocity, speed_limit_ok=speed_ok,
                  position_tolerance_m=0.002, orientation_tolerance_deg=2.0,
                  model=str(args.model.resolve()), coordinates='MuJoCo world metres; hand-base site w,x,y,z',
                  physics_stepping=False, collision_avoidance=False, grasping=False, stereo_input=False,
                  limitations='Kinematic fixed-base arm test, not a dynamics/torque/hardware controller or collision-free planner. Local IK failure does not prove a target is globally unreachable.')
    np.savez_compressed(args.output/'ik_trajectory.npz', timestamps=np.array(times), qpos=poses,
                        sides=np.array(sides), target_position=np.asarray(target_positions).reshape(-1, len(sides), 3),
                        target_quaternion_wxyz=np.asarray(target_quaternions).reshape(-1, len(sides), 4))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    for side, summary in summaries.items():
        final = summary['final']
        if final:
            print(f'{side}: {final["status"]}; position={1000*final["position_error_m"]:.3f} mm, '
                  f'orientation={final["orientation_error_deg"]:.3f} deg', flush=True)
        else:
            print(f'{side}: NO_FRAMES', flush=True)
    print(f'Completed={complete}; fixed joints={unchanged}; limits={bounds_ok}; max speed={max_speed:.3f} rad/s; success={success}', flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
