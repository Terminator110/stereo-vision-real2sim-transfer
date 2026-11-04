"""Marker-first replay of a reconstructed single-tag shared-workspace trajectory."""
import csv
import json
from pathlib import Path
import threading
import time

import mujoco
import numpy as np

from g1_hand_ik import HandIKController, HandTarget, quaternion
from g1_workspace import CollisionChecker, load_workspace
from select_g1_workspace import validate_path
from trajectory_io import load_xyz, scene_table_frame, sha256


def replay_input(path, model):
    path = Path(path)
    trajectory = load_xyz(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if (metadata.get('coordinate_frame') != 'mujoco_world' or metadata.get('mapping_mode') != 'shared_manipulation_workspace'
            or metadata.get('units') != 'metres' or metadata.get('scale') != 1
            or metadata.get('hand_base_offset_m') != [0, 0, 0]
            or metadata.get('selected_tag_id') not in (2, 4, 5)
            or metadata.get('csv_sha256') != sha256(path)):
        raise ValueError('Require matching reconstructed world CSV metadata, single tag, metres, scale 1 and zero offset')
    workspace = Path(metadata['workspace_profile'])
    if sha256(workspace) != metadata['workspace_sha256']:
        raise ValueError('Workspace changed: explicitly re-export the world trajectory and review it again')
    rotation, origin, profile = load_workspace(workspace)
    _, centre = scene_table_frame(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'work_table_top')
    local = (origin-centre) @ data.geom_xmat[table].reshape(3, 3)
    if abs(local[2]) > 1e-6 or np.any(np.abs(local[:2]) > model.geom_size[table, :2]+1e-6):
        raise ValueError('Shared origin must lie on the existing tabletop surface')
    source = Path(metadata['source_table_csv'])
    if sha256(source) != metadata['source_table_sha256']:
        raise ValueError('Reconstructed source trajectory changed')
    original = load_xyz(source)
    if not np.allclose(trajectory.points, original.points @ rotation.T+origin, atol=1e-10) or not np.array_equal(trajectory.times, original.times):
        raise ValueError('World trajectory altered displacement scale, timing, or workspace mapping')
    raw_path = Path(metadata['raw_table_csv'])
    if sha256(raw_path) != metadata['raw_table_csv_sha256']:
        raise ValueError('Raw source coordinates changed')
    raw = load_xyz(raw_path).points @ rotation.T+origin
    probes = []
    scan = workspace.with_name(workspace.stem+'_reachability.csv')
    if scan.is_file():
        with scan.open() as handle:
            for row in csv.DictReader(handle):
                if np.allclose([float(row['origin_x']), float(row['origin_y'])], origin[:2]):
                    probes.append((np.array([float(row[k]) for k in ('x', 'y', 'z')]), row['reachable'] == 'True'))
    return trajectory, metadata, rotation, origin, raw, probes


def draw_overlay(scene, rotation, origin, raw, processed, probes, marker):
    def sphere(point, radius, color):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.full(3, radius), point, np.eye(3).ravel(), np.asarray(color))
        scene.ngeom += 1
    def connector(a, b, color, arrow=False):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.zeros(3), a, np.eye(3).ravel(), np.asarray(color))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE,
                            .004 if arrow else .001, a, b)
        scene.ngeom += 1
    sphere(origin, .012, [1, 1, 1, 1])
    for axis, color in enumerate(([1, .1, .1, 1], [.1, 1, .1, 1], [.1, .3, 1, 1])):
        connector(origin, origin+rotation[:, axis]*.10, color, True)
    for point, reachable in probes:
        sphere(point, .003, [.1, .8, .2, .55] if reachable else [.9, .1, .1, .4])
    for point in raw:
        sphere(point, .0035, [.7, .7, .7, .7])
    for a, b in zip(processed[:-1], processed[1:]):
        connector(a, b, [.1, .6, 1, .85])
    if marker is not None:
        sphere(marker, .011, [1, .65, .05, 1])


def trajectory_start_pose(controller, checker, point, orientation):
    """Prepare a checked initial configuration, NOT a motion from the neutral pose."""
    arm = controller.arms['right']
    result = arm.solve(controller.qpos, HandTarget(point, orientation))
    if not result.converged:
        raise ValueError(f'First trajectory pose has no accepted IK solution: {result.position_error_m:.4f} m error')
    hits = checker.check(result.candidate_qpos)
    if hits:
        raise ValueError(f'First trajectory pose collides: {hits[:3]}')
    return result.candidate_qpos.copy(), dict(
        mode='trajectory_start_configuration', simulation_only=True,
        initialization_max_joint_change_rad=float(np.max(np.abs(result.candidate_qpos-controller.qpos))),
        initialization_is_motion=False, skipped_neutral_to_start_transition=True,
        first_pose_position_error_m=result.position_error_m,
        first_pose_orientation_error_deg=result.orientation_error_deg)


def replay(args):
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if getattr(args, 'hide_reachability', False):
        cloud = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'g1_right_wrist_reachable_samples')
        if cloud >= 0:
            model.site_rgba[model.site_bodyid == cloud, 3] = 0
    trajectory, metadata, rotation, origin, raw, probes = replay_input(args.trajectory, model)
    if abs(trajectory.times[0]) > 1e-8:
        raise ValueError('Export a continuous segment starting at time zero; no implicit timeline shifting')
    execute = args.execute_trajectory
    if execute and not args.workspace_reviewed:
        raise ValueError('Run marker preview first, then explicitly supply --workspace-reviewed to move the arm')
    gaps = trajectory.gaps(args.max_trajectory_gap)
    if execute and gaps:
        raise ValueError('IK replay refuses disconnected tracking gaps; export one supported continuous segment')
    waist_yaw = getattr(args, 'waist_yaw', False)
    controller = HandIKController(model, ('right',), args.max_velocity, include_waist_yaw=waist_yaw)
    arm = controller.arms['right']
    _, start_rotation = arm.hand_pose(controller.qpos)
    orientation = quaternion(start_rotation)
    checker = CollisionChecker(model, include_waist_yaw=waist_yaw) if execute else None
    args.output.mkdir(parents=True, exist_ok=False)
    initialization = dict(mode='neutral_then_preposition' if execute else 'marker_only', initialization_is_motion=execute)
    initial_pose = controller.qpos.copy()
    if args.start_at_trajectory:
        initial_pose, initialization = trajectory_start_pose(controller, checker, trajectory.points[0], orientation)
    preflight = dict(ok=None, note='Marker-only preview; arm does not move')
    if execute:
        preflight, candidates = validate_path(arm, checker, trajectory.points, orientation, initial_pose)
        if preflight['ok']:
            required = np.abs(np.diff(np.array(candidates)[:, arm.addresses], axis=0))/np.diff(trajectory.times)[:, None]*args.speed
            preflight['peak_candidate_joint_speed_rad_s'] = float(required.max())
        (args.output/'preflight.json').write_text(json.dumps(preflight, indent=2)+'\n')
        if not preflight['ok']:
            raise ValueError(f'Workspace preflight failed before arm motion: {preflight}; see {args.output}')
    if args.start_at_trajectory:
        controller.qpos[:] = initial_pose
    data = mujoco.MjData(model)
    data.qpos[:] = controller.qpos
    mujoco.mj_forward(model, data)
    viewer, threads = None, []
    poses, phases, stamps, rows = [], [], [], []
    ticks, max_speed, next_render = 0, 0.0, 0.0
    dt = 1/args.hz
    completed, prepositioned, failure = False, not execute or args.start_at_trajectory, None
    start_wall = time.monotonic()
    print(f'{"IK REPLAY" if execute else "MARKER ONLY"}: Tag {metadata["selected_tag_id"]}; '
          f'origin={origin.tolist()}, yaw={metadata["yaw_degrees"]} deg, scale=1, speed={args.speed}', flush=True)
    print('White=workspace origin; RGB=axes; grey=raw; blue=reconstructed; orange=moving target; green/red=sampled reachability.', flush=True)
    print('All arm commands get collision and joint-limit checks; marker-only preview does not certify reachability.', flush=True)
    if waist_yaw:
        print('IK includes right arm + waist yaw only. Collision checks include both arms and torso; other joint angles stay fixed.', flush=True)
    if args.start_at_trajectory:
        print('Simulation initialized at the first checked trajectory pose. No neutral-to-start motion is executed; replay speed limits remain active.', flush=True)

    def frame(phase, recorded_time, point):
        nonlocal ticks, max_speed, next_render
        if viewer is not None and not viewer.is_running():
            return False, None
        before = controller.qpos.copy()
        state = dict(status='MARKER_ONLY' if point is not None else 'GAP_NO_TARGET', reached=False)
        if execute:
            pose, states = controller.step({'right': HandTarget(point, orientation)}, dt)
            state = states['right']
            hits = checker.transition(before, pose, arm.addresses)
            if hits:
                controller.qpos[:] = before
                state = dict(status='COLLISION_HOLD', reached=False, collisions=hits[:3])
        pose = controller.qpos.copy()
        max_speed = max(max_speed, float(np.max(np.abs(pose-before))/dt))
        poses.append(pose)
        phases.append(phase)
        stamps.append(recorded_time if recorded_time is not None else np.nan)
        rows.append(dict(phase=phase, source_time_s=recorded_time, **state))
        data.qpos[:] = pose
        data.time = ticks*dt
        mujoco.mj_forward(model, data)
        if viewer is not None:
            elapsed = time.monotonic()-start_wall
            if elapsed >= next_render:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    # Render each connected piece independently, never a line through a large gap.
                    cuts = [0, *[i+1 for i in gaps], len(trajectory.times)]
                    draw_overlay(viewer.user_scn, rotation, origin, raw, np.empty((0, 3)), probes, point)
                    for a, b in zip(cuts[:-1], cuts[1:]):
                        # No repeated origins/probes: draw path through a temporary empty raw/probe set.
                        for p in trajectory.points[a:b: max(1, (b-a)//100)]:
                            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, .0025), p, np.eye(3).ravel(), np.array([.1, .6, 1, .8]))
                                viewer.user_scn.ngeom += 1
                viewer.sync()
                next_render = elapsed+1/30
            ticks += 1
            time.sleep(max(0, start_wall+ticks*dt-time.monotonic()))
        return True, state

    try:
        if not args.headless:
            from mujoco import viewer as mj_viewer
            before = set(threading.enumerate())
            viewer = mj_viewer.launch_passive(model, data)
            threads = [t for t in threading.enumerate() if t not in before]
            viewer.cam.lookat[:] = [.35, -.07, .90]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 135, -23
            start_wall = time.monotonic()
        if execute and not args.start_at_trajectory:
            for _ in range(int(args.preposition_seconds*args.hz)):
                ok, state = frame('PREPOSITION', None, trajectory.points[0])
                if not ok:
                    break
                if state['status'] in ('COLLISION_HOLD', 'IK_FAILED_HOLD'):
                    failure = state
                    break
                if state['reached']:
                    prepositioned = True
                    break
            if not prepositioned and failure is None:
                failure = dict(status='PREPOSITION_NOT_REACHED')
        if prepositioned:
            duration = trajectory.times[-1]/args.speed
            for index in range(int(np.ceil(duration*args.hz))+1):
                source_time = min(index/args.hz*args.speed, trajectory.times[-1])
                point = trajectory.sample(source_time, args.max_trajectory_gap)
                ok, state = frame('REPLAY' if execute else 'PREVIEW', source_time, point)
                if not ok:
                    break
                if execute and state['status'] in ('COLLISION_HOLD', 'IK_FAILED_HOLD'):
                    failure = state
                    break
            else:
                completed = True
    except KeyboardInterrupt:
        failure = dict(status='INTERRUPTED')
    finally:
        if viewer is not None:
            viewer.close()
            for thread in threads:
                thread.join(timeout=5)
    poses = np.asarray(poses).reshape(-1, model.nq)
    fixed = np.setdiff1d(np.arange(model.nq), arm.addresses)
    unchanged = bool(not len(poses) or np.all(poses[:, fixed] == controller.initial_qpos[fixed]))
    bounds_ok = bool(not len(poses) or (np.all(poses[:, arm.addresses] >= arm.lower-1e-9) and np.all(poses[:, arm.addresses] <= arm.upper+1e-9)))
    all_static = bool(not len(poses) or np.all(poses == controller.initial_qpos))
    success = completed and failure is None and unchanged and bounds_ok and max_speed <= args.max_velocity+1e-8
    tracked = [row for row in rows if row['phase'] == 'REPLAY']
    max_tracking = max((row['position_error_m'] for row in tracked if 'position_error_m' in row), default=None)
    max_orientation = max((row['orientation_error_deg'] for row in tracked if 'orientation_error_deg' in row), default=None)
    if execute:
        success = success and bool(tracked and all(row['reached'] for row in tracked))
    else:
        success = success and all_static
    report = dict(success=bool(success), executed_arm=execute, completed=completed, failure=failure,
                  waist_yaw_enabled=waist_yaw, controlled_joints=arm.joint_names,
                  initialization=initialization,
                  preview_all_joints_static=all_static if not execute else None,
                  other_joints_unchanged=unchanged, joint_limits_ok=bounds_ok,
                  peak_commanded_joint_speed_rad_s=max_speed, joint_speed_limit_rad_s=args.max_velocity,
                  speed=args.speed, recorded_duration_s=float(trajectory.times[-1]),
                  nominal_replay_duration_s=float(trajectory.times[-1]/args.speed), wall_duration_s=time.monotonic()-start_wall,
                  max_replay_position_error_m=max_tracking, max_replay_orientation_error_deg=max_orientation,
                  timing_note='Source timestamps preserved; headless replay is accelerated. Native viewer is paced to source time/speed, plus optional separately logged prepositioning.',
                  gaps_not_bridged=gaps, selected_tag_id=metadata['selected_tag_id'],
                  source_segment=metadata['selected_segment'], full_approach=metadata['all_approach_reconstructed'],
                  workspace_origin_world_m=origin.tolist(), yaw_degrees=metadata['yaw_degrees'], scale=1,
                  source_origin='calibrated v3 origin, not physical table centre', preflight=preflight,
                  collision_checks=checker.description() if checker else dict(note='Marker-only preview: collision validation not performed'),
                  final_state=rows[-1] if rows else None,
                  unchanged_initial_hand_orientation_wxyz=orientation.tolist(), hand_base_offset_m=[0, 0, 0],
                  physics_stepping=False, limits='Discrete kinematic collision checking, not dynamics, torque control, a general motion planner, or certified hardware safety')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'replay_states.json').write_text(json.dumps(rows, indent=2)+'\n')
    np.savez_compressed(args.output/'replay.npz', qpos=poses, phase=np.array(phases), source_time_s=np.array(stamps),
                        initial_qpos=initial_pose, model_initial_qpos=controller.initial_qpos)
    print(f'Success={success}; completed={completed}; other joints fixed={unchanged}; peak joint speed={max_speed:.3f} rad/s. Report: {args.output}', flush=True)
    return 0 if success else 1
