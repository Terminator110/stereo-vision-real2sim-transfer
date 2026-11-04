#!/usr/bin/env python3
"""Dynamic DEX5 replay from sim_qpos.npz (no hardcoded grasp optimizer).

Creates new XML copies, drives recorded robot joint positions using position
actuators and mj_step, and optionally replaces the post-grasp movement with
an IK-generated vertical wrist lift. Cup is free and moves only via contacts.
"""
import argparse
import csv
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def find_include(xmlroot, suffix):
    found = [node for node in xmlroot.findall('include')
             if Path(node.get('file', '')).name == suffix]
    if len(found) != 1:
        raise RuntimeError(f'Expected one <include> for {suffix}, found {len(found)}')
    return found[0]


def source_of_include(node, referring_xml):
    p = Path(node.get('file'))
    if not p.is_absolute():
        p = referring_xml.parent / p
    return p.resolve()


def make_dynamic_scene(root, out, mass, friction, finger_torque, timestep,
                       cup_xy, cup_base_z):
    original = root / 'models/scenario_g1_dex5_stationary.xml'
    original_scene = root / 'models/scenes/table_and_cup.xml'
    if not original.exists() or not original_scene.exists():
        raise FileNotFoundError('Original MuJoCo scenario or table_and_cup.xml missing')
    reference = mujoco.MjModel.from_xml_path(str(original))
    main = ET.parse(original)
    mainroot = main.getroot()

    # Copy the exact original environment; change only the cup to a free body
    # and add physical collision surfaces. The visual geometry remains intact.
    scene = ET.parse(original_scene)
    scene_root = scene.getroot()
    cup = scene_root.find('.//body[@name="table_cup"]')
    if cup is None:
        raise RuntimeError('Cup body table_cup not found in scene')
    if cup.find('freejoint') is not None or cup.find('joint') is not None:
        raise RuntimeError('Source cup is already jointed; refusing double joint')
    visual_radius, height = 0.041, 0.098  # overwritten from original mesh below

    static = mujoco.MjData(reference)
    mujoco.mj_forward(reference, static)
    cup_visual_id = mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_GEOM, 'table_cup_visual')
    if cup_visual_id < 0:
        raise RuntimeError('Cannot find table_cup_visual geometry')
    visual = reference.geom(cup_visual_id)
    if reference.geom_type[cup_visual_id] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(reference.geom_dataid[cup_visual_id])
        start = int(reference.mesh_vertadr[mesh_id]); count = int(reference.mesh_vertnum[mesh_id])
        verts = reference.mesh_vert[start:start+count] * np.asarray(reference.geom_size[cup_visual_id, :3])
        # Reported physical dimensions are 82 mm x 98 mm, validated elsewhere;
        # mesh radii can be asset-scale dependent. Use scene-derived geom scale
        # only when bounds are plausible.
        dimensions = np.ptp(verts, axis=0)
        if 0.04 < max(dimensions[0], dimensions[1]) < 0.15 and 0.04 < dimensions[2] < 0.2:
            visual_radius = max(dimensions[0], dimensions[1]) / 2
            height = dimensions[2]
    cup.set('pos', f'{cup_xy[0]:.9f} {cup_xy[1]:.9f} {cup_base_z:.9f}')
    ET.SubElement(cup, 'freejoint', name='test_cup_freejoint')
    mass = float(mass)
    ix = mass * (3*visual_radius**2 + height**2)/12
    iz = mass * visual_radius**2/2
    ET.SubElement(cup, 'inertial', pos=f'0 0 {height/2}', mass=str(mass),
                  diaginertia=f'{ix} {ix} {iz}')
    common = dict(contype='2', conaffinity='1', density='0',
                  friction=f'{friction} .005 .0001', condim='4',
                  solref='.01 1', solimp='.95 .99 .001',
                  group='5', rgba='0 0 0 0')
    wall = .002
    for i in range(32):
        a = 2*math.pi*i/32
        bottom = (visual_radius-.0042-wall)*np.array([math.cos(a), math.sin(a)])
        top = (visual_radius-wall)*np.array([math.cos(a), math.sin(a)])
        ET.SubElement(cup, 'geom', name=f'test_cup_wall_{i:02d}',
                      type='capsule', size=str(wall),
                      fromto=f'{bottom[0]} {bottom[1]} {wall} '
                             f'{top[0]} {top[1]} {height-wall}', **common)
    ET.SubElement(cup, 'geom', name='test_cup_bottom', type='cylinder',
                  pos='0 0 .0015', size=f'{visual_radius-.0042} .0015', **common)
    scene_output = out / 'dynamic_table_and_cup.xml'
    scene.write(scene_output, encoding='unicode')

    robot_node = find_include(mainroot, 'g1_dex5_stationary.xml')
    robot_source = source_of_include(robot_node, original)
    robot = ET.parse(robot_source)
    rroot = robot.getroot()
    actuator = rroot.find('actuator')
    if actuator is None:
        actuator = ET.SubElement(rroot, 'actuator')
    else:
        actuator.clear()
    for jid in range(reference.njnt):
        name = reference.joint(jid).name
        if reference.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        is_finger = '_dex5_' in name
        is_wrist = '_wrist_' in name
        kp, kv = ((6., .12) if is_finger else ((70., 4.) if is_wrist else (350., 25.)))
        force = finger_torque if is_finger else (25. if is_wrist else 90.)
        low, high = reference.jnt_range[jid]
        if high <= low:
            # This environment should use limited hinge joints.
            raise RuntimeError(f'Unlimited or zero-range hinge joint: {name}')
        ET.SubElement(actuator, 'position', name='physics_position_'+name,
                      joint=name, kp=str(kp), kv=str(kv),
                      ctrllimited='true', ctrlrange=f'{low} {high}',
                      forcelimited='true', forcerange=f'{-force} {force}')
    robot_output = out / 'dynamic_robot_actuated.xml'
    robot.write(robot_output, encoding='unicode')
    robot_node.set('file', str(robot_output))
    scene_node = find_include(mainroot, 'table_and_cup.xml')
    scene_node.set('file', str(scene_output))
    option = mainroot.find('option')
    if option is None:
        option = ET.SubElement(mainroot, 'option')
    option.set('timestep', str(timestep))
    option.set('gravity', '0 0 -9.81')
    option.set('iterations', '100')
    option.set('integrator', 'implicitfast')
    output = out / 'scenario_dynamic_replay.xml'
    main.write(output, encoding='unicode')
    return output, visual_radius, height



def cup_coordinates_from_reference(reference_file, model_file):
    """Load the detected static cup position in MuJoCo world coordinates."""
    ref = json.loads(reference_file.read_text())
    cup = ref.get("objects", {}).get("cup", {})

    if not cup.get("valid", False):
        raise RuntimeError(f"Cup localization invalid: {reference_file}")

    table_xy = np.asarray(cup["table_xy_m"], dtype=float)
    world_xy = np.asarray(cup["world_xy_m"], dtype=float)
    origin = np.asarray(ref["world_origin_m"], dtype=float)

    # Verify the camera-table-to-world transformation.
    yaw = math.radians(float(ref["table_yaw_deg"]))
    rot = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw),  math.cos(yaw)]
    ])

    expected_xy = origin[:2] + rot @ table_xy

    if not (
        np.isfinite(table_xy).all()
        and np.isfinite(world_xy).all()
        and np.isfinite(origin).all()
    ):
        raise RuntimeError("Nonfinite cup coordinates")

    if np.linalg.norm(expected_xy - world_xy) > 0.005:
        raise RuntimeError("Cup table/world coordinate mismatch")

    # Confirm the table height matches the actual MuJoCo scene.
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tableid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "work_table_top"
    )

    if tableid < 0:
        raise RuntimeError("Missing work_table_top geometry")

    base_z = float(
        data.geom_xpos[tableid, 2] + model.geom_size[tableid, 2]
    )

    if abs(base_z - origin[2]) > 0.005:
        raise RuntimeError(
            f"Table height mismatch: scene={base_z}, reference={origin[2]}"
        )

    print("========== CUP WORLD REFERENCE ==========", flush=True)
    print("Table XY:", table_xy, flush=True)
    print("World XY:", world_xy, flush=True)
    print("Base Z:", base_z, flush=True)

    return table_xy, world_xy, base_z


def cup_coordinates(root, objects_file, model_file):
    from reconstruct_grasp_closure import YAW, ORIGIN
    rows = list(csv.DictReader(objects_file.open()))
    early = [r for r in rows if int(r['frame']) <= 21 and
             r.get('cup_valid_tabletop') == '1' and
             r.get('cup_x_table_m') and r.get('cup_y_table_m')]
    if not early:
        early = [r for r in rows if r.get('cup_valid_tabletop') == '1'
                 and r.get('cup_x_table_m') and r.get('cup_y_table_m')]
    if not early:
        raise RuntimeError('No valid cup tabletop localization in objects_keyframes.csv')
    xy = np.median(np.array([[float(r['cup_x_table_m']),float(r['cup_y_table_m'])]
                             for r in early]), axis=0)
    point = np.asarray(YAW, float) @ np.r_[xy, 0.]
    world = point + np.asarray(ORIGIN, float)
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    tableid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'work_table_top')
    if tableid < 0:
        raise RuntimeError('No work_table_top collision geom')
    z = float(data.geom_xpos[tableid,2]+model.geom_size[tableid,2])
    return xy, world[:2], z



def palm_aligned_cup_xy(model_file, pose, distance=0.075, sideways=0.0):
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    data.qpos[:len(pose)] = pose
    mujoco.mj_forward(model, data)

    palm = data.xpos[model.body("g1_01_dex5_right_palm").id].copy()

    knuckles = np.array([
        data.xpos[
            model.body(f"g1_01_dex5_right_Link_{n}2R").id
        ].copy()
        for n in range(2, 6)
    ])

    direction = (knuckles.mean(axis=0) - palm)[:2]
    norm = np.linalg.norm(direction)

    if norm < 0.015:
        raise RuntimeError("Cannot determine palm-forward direction")

    direction /= norm
    lateral = np.array([-direction[1], direction[0]])

    xy = palm[:2] + distance * direction + sideways * lateral

    print("Palm XYZ:", palm)
    print("Cup target XY:", xy)
    return xy


def find_arm(model):
    try:
        from dex5_video_to_sim_side_by_side import ARM
        names = list(ARM)
    except ImportError:
        names = []
    if not names:
        names = [model.joint(i).name for i in range(model.njnt)
                 if 'right' in model.joint(i).name.lower()
                 and '_dex5_' not in model.joint(i).name.lower()]
    ids = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,name)
        if jid >= 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            ids.append(int(jid))
    if len(ids)<4:
        raise RuntimeError(f'Could not identify sufficient right arm joints: {names}')
    return np.array([int(model.jnt_qposadr[i]) for i in ids]), np.array([int(model.jnt_dofadr[i]) for i in ids]), np.asarray(model.jnt_range[ids]), [model.joint(i).name for i in ids]


def lift_targets(model, source_pose, robot_nq, height, siteid, count=41):
    data=mujoco.MjData(model)
    data.qpos[:robot_nq]=source_pose
    mujoco.mj_forward(model,data)
    arm_q, arm_v, limits, names=find_arm(model)
    start_pos=data.site_xpos[siteid].copy()
    start_rot=data.site_xmat[siteid].reshape(3,3).copy()
    goals=[]
    q=source_pose.copy()
    failures=[]
    for fraction in np.linspace(0, 1, count):
        goal=start_pos+np.array([0.,0.,height*fraction])
        for it in range(130):
            data.qpos[:robot_nq]=q
            mujoco.mj_forward(model,data)
            dp=goal-data.site_xpos[siteid]
            now=data.site_xmat[siteid].reshape(3,3)
            delta=start_rot@now.T
            dr=.5*np.array([delta[2,1]-delta[1,2],delta[0,2]-delta[2,0],delta[1,0]-delta[0,1]])
            if np.linalg.norm(dp)<.0007 and np.linalg.norm(dr)<.012:
                break
            jp=np.zeros((3,model.nv));jr=np.zeros((3,model.nv))
            mujoco.mj_jacSite(model,data,jp,jr,siteid)
            A=np.vstack([jp[:,arm_v],.4*jr[:,arm_v]])
            b=np.r_[dp,.4*dr]
            dq=A.T@np.linalg.solve(A@A.T+1e-3*np.eye(6),b)
            q[arm_q]=np.clip(q[arm_q]+np.clip(dq,-.04,.04),limits[:,0]+1e-5,limits[:,1]-1e-5)
        if np.linalg.norm(dp)>.01:
            failures.append([float(fraction),float(np.linalg.norm(dp))])
        goals.append(q.copy())
    return np.asarray(goals),failures,names


def contact_metrics(model,data,cup_id):
    cup_contacts=0;finger_contacts=0;thumb_contacts=0;max_penetration=0.;table_contacts=0
    for i in range(data.ncon):
        c=data.contact[i]
        b1=int(model.geom_bodyid[c.geom1]);b2=int(model.geom_bodyid[c.geom2])
        if cup_id not in (b1,b2):continue
        other=int(c.geom2 if b1==cup_id else c.geom1)
        other_body=model.body(int(model.geom_bodyid[other])).name
        cup_contacts+=1
        max_penetration=max(max_penetration,max(0.,-float(c.dist)))
        if '_dex5_' in other_body:
            finger_contacts+=1
            if 'Link_1' in other_body:thumb_contacts+=1
        if model.geom(other).name=='work_table_top':table_contacts+=1
    return cup_contacts,finger_contacts,thumb_contacts,table_contacts,max_penetration


def run(args):
    root=args.root.resolve()
    take=args.take
    src=(args.trajectory.expanduser().resolve() if args.trajectory else root/'outputs'/f'smoothed_{take}'/'sim_qpos.npz')
    objects=root/'code/stereo/recordings_three_cam'/take/'processed/object_localization/objects_keyframes.csv'
    if not src.is_file() or not objects.is_file():
        raise FileNotFoundError(f'Missing input: {src if not src.exists() else objects}')
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    args.output.mkdir(parents=True)
    # Preserve relative MuJoCo asset paths from the original models directory.
    for source in (root / 'models').iterdir():
        destination = args.output / source.name
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(source, target_is_directory=source.is_dir())
    with np.load(src) as saved:
        qs=saved['qpos'].copy()
        times=saved['time_s'].copy()
        frames=saved['source_frames'].copy()

    placement_frame = None

    if args.cup_from_approach_blue:
        ann = (
            root / "code/stereo/recordings_three_cam"
            / take / "phase_annotations.csv"
        )

        with ann.open(newline="") as f:
            phases = {
                r["phase"]: r
                for r in csv.DictReader(f)
            }

        placement_frame = int(phases["approach"]["end_frame"])
        args.grasp_frame = int(phases["grasp"]["end_frame"])

        assert placement_frame < args.grasp_frame

        if not np.any(frames == placement_frame):
            raise RuntimeError("Approach frame missing")

        print(
            "Palm pose frame:", placement_frame,
            "Grasp end:", args.grasp_frame,
            flush=True
        )

    matches=np.where(frames==args.grasp_frame)[0]
    if len(matches)!=1:
        raise RuntimeError(f'Grasp frame {args.grasp_frame} missing from {src}')
    stop=int(matches[0]);robot_nq=qs.shape[1]
    original=root/'models/scenario_g1_dex5_stationary.xml'
    # Use measured cup position from this recording.
    # Reject previous manual/palm overrides.
    if getattr(args, "cup_at_palm", False) or getattr(args, "fixed_cup_table", False):
        raise ValueError("Manual cup overrides disabled for world-reference replay")

    ref_file = objects.with_name("object_world_reference.json")

    if not ref_file.is_file():
        raise FileNotFoundError(ref_file)

    xy_table,xy_world,z = cup_coordinates_from_reference(
        ref_file, original
    )

    localized_world_xy=np.asarray(xy_world,float).copy()
    path,radius,height=make_dynamic_scene(root,args.output,args.mass_kg,args.friction,
                                           args.finger_torque,args.timestep,xy_world,z)

    if args.cup_from_approach_blue:
        ref = mujoco.MjModel.from_xml_path(str(original))
        rd = mujoco.MjData(ref)

        idx = int(np.flatnonzero(frames == placement_frame)[0])
        rd.qpos[:] = qs[idx]
        mujoco.mj_forward(ref, rd)

        marker_xyz, marker_R = _dex5_palm_pose(
            ref, rd, args.palm_marker_offset_mm
        )

        blue = marker_R[:, 2]

        centre_distance = float(
            radius + args.cup_clearance_mm / 1000.
        )

        if centre_distance <= radius:
            raise ValueError("Cup clearance must be positive")

        # Project the blue axis onto the horizontal table plane.
        blue_flat = np.array([blue[0], blue[1], 0.0])

        norm = np.linalg.norm(blue_flat)
        if norm < 1e-6:
            raise RuntimeError("Blue axis is nearly vertical")

        blue_flat /= norm

        # Move horizontally from the palm marker.
        centre_xyz = marker_xyz + blue_flat * centre_distance

        # Keep the cup upright and resting on the table.
        centre_xyz[2] = z + height / 2.0
        base_xyz = centre_xyz - np.array([0., 0., height / 2.])

        xml = args.output / "dynamic_table_and_cup.xml"
        tree = ET.parse(xml)

        cup = tree.find('.//body[@name="table_cup"]')
        if cup is None:
            raise RuntimeError("Cup body missing")

        cup.set(
            "pos",
            " ".join(f"{v:.9f}" for v in base_xyz)
        )

        tree.write(xml, encoding="unicode")

        xy_world = centre_xyz[:2].copy()

        record = dict(
            approach_end_frame=placement_frame,
            grasp_end_frame=args.grasp_frame,
            marker_x_angle_deg=-30.0,
            marker_xyz_m=marker_xyz.tolist(),
            blue_world=blue_flat.tolist(),
            cup_radius_m=float(radius),
            clearance_m=args.cup_clearance_mm / 1000.,
            cup_centre_xyz_m=centre_xyz.tolist(),
            cup_base_xyz_m=base_xyz.tolist(),
            base_minus_table_m=float(base_xyz[2] - z)
        )

        (args.output / "cup_placement.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )

        print("CUP PLACEMENT:")
        print(json.dumps(record, indent=2), flush=True)

        if abs(base_xyz[2] - z) > .005:
            print(
                "WARNING: Cup base is not at table height. "
                "Gravity may move it before grasp.",
                flush=True
            )

    model=mujoco.MjModel.from_xml_path(str(path))
    data=mujoco.MjData(model)
    if model.nq!=robot_nq+7:
        raise RuntimeError(f'Robot/Cup qpos mismatch: source {robot_nq}, model {model.nq}')
    free=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'test_cup_freejoint')
    addr=int(model.jnt_qposadr[free]);cup_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'table_cup')
    if addr!=robot_nq:
        raise RuntimeError(f'Cup free joint address {addr} != source DOFs {robot_nq}')
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'g1_01_right_wrist_ik_site')
    if site<0:raise RuntimeError('Missing g1_01_right_wrist_ik_site')
    ctrls=np.array([int(model.jnt_qposadr[int(model.actuator_trnid[i,0])]) for i in range(model.nu)])
    if np.any(ctrls>=robot_nq):raise RuntimeError('Cup must have no actuators')
    data.qpos[:robot_nq]=qs[0]
    # qpos0 initializes cup freejoint at its declared body position.
    data.qpos[addr:addr+7]=model.qpos0[addr:addr+7]
    data.ctrl[:]=data.qpos[ctrls]
    mujoco.mj_forward(model,data)
    init_cup=data.xpos[cup_id].copy()
    wrist_grasp_data=mujoco.MjData(model)
    wrist_grasp_data.qpos[:robot_nq]=qs[stop]
    mujoco.mj_forward(model,wrist_grasp_data)
    wrist_grasp_z=float(wrist_grasp_data.site_xpos[site,2])
    d=float(np.linalg.norm(wrist_grasp_data.site_xpos[site]-(init_cup+np.array([0,0,height/2]))))
    print(f'Cup tabletop XY: {xy_table}; mapped world XY: {xy_world}; Z={z:.4f}',flush=True)
    print(f'Grasp frame: {args.grasp_frame}, wrist-to-cup-center: {d*1000:.1f} mm',flush=True)
    if d>.24:
        print('WARNING: Wrist far from cup; validate coordinate mapping before trusting contacts.',flush=True)
    lift_q,failures,arm_names=lift_targets(model,qs[stop],robot_nq,args.lift_height,site)
    print('Lift IK arm:',arm_names,flush=True)
    print('Lift IK targets over 10 mm position error:',failures[:5],flush=True)
    if failures:
        print(f'WARNING: {len(failures)}/{len(lift_q)} lift targets have >10 mm IK error',flush=True)
    # Allow the free cup to settle before the recorded approach.
    settle=.4
    approach=float(times[stop]-times[0])
    grasp_hold=.6
    lift_duration=1.5
    hold_duration=1.0
    total=settle+approach+grasp_hold+lift_duration+hold_duration
    every=max(1,round(.02/args.timestep))
    trajectories=[];records=[]
    times_recorded=times[:stop+1]-times[0]
    total_steps=int(math.ceil(total/args.timestep))
    print(f'Simulating {total_steps} physics steps; recording through frame {args.grasp_frame} then 60 mm lift',flush=True)
    for step in range(total_steps):
        if step and step%2000==0:
            print(f'Physics progress: {step}/{total_steps} steps',flush=True)
        simtime=step*args.timestep
        if simtime<settle:
            phase='settle';target=qs[0]
        elif simtime<settle+approach:
            phase=('grasp' if args.cup_from_approach_blue
                   and simtime >= settle + float(
                       times[np.flatnonzero(frames==placement_frame)[0]]
                       - times[0])
                   else 'approach');t=simtime-settle
            k=int(np.clip(np.searchsorted(times_recorded,t,side='right')-1,0,stop-1))
            alpha=float(np.clip((t-times_recorded[k])/max(1e-8,times_recorded[k+1]-times_recorded[k]),0,1))
            target=(1-alpha)*qs[k]+alpha*qs[k+1]
        elif simtime<settle+approach+grasp_hold:
            phase='grasp_hold';target=qs[stop]
        elif simtime<settle+approach+grasp_hold+lift_duration:
            phase='lift';frac=(simtime-settle-approach-grasp_hold)/lift_duration
            x=frac*(len(lift_q)-1);i=min(int(x),len(lift_q)-2)
            target=(1-(x-i))*lift_q[i]+(x-i)*lift_q[i+1]
        else:
            phase='hold';target=lift_q[-1]
        data.ctrl[:]=np.clip(target[ctrls],model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f'Physics instability at step {step}; inspect controllers and timestep')
        if step%every==0:
            contacts,fingers,thumb,table,penetration=contact_metrics(model,data,cup_id)
            wrist_err=float(np.linalg.norm(data.site_xpos[site]-wrist_grasp_data.site_xpos[site]))
            cup_rise=float(data.xpos[cup_id,2]-init_cup[2])
            row=dict(time_s=float(data.time),phase=phase,cup_rise_m=cup_rise,
                     cup_xy_shift_m=float(np.linalg.norm(data.xpos[cup_id,:2]-init_cup[:2])),
                     cup_contacts=contacts,finger_contacts=fingers,thumb_contacts=thumb,
                     table_contacts=table,max_cup_penetration_m=penetration,
                     wrist_displacement_from_grasp_m=wrist_err,
                     wrist_lift_world_z_m=float(data.site_xpos[site,2]-wrist_grasp_z))
            records.append(row);trajectories.append(data.qpos.copy())
    with (args.output/'physics_trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    np.savez_compressed(args.output/'physics_replay.npz',qpos=np.asarray(trajectories),
                        time_s=np.array([r['time_s'] for r in records]))
    hold=[r for r in records if r['phase']=='hold']
    report=dict(source=str(src),object_localization=str(objects),grasp_frame=args.grasp_frame,
                planned_lift_m=args.lift_height,grasp_wrist_to_cup_center_m=d,
                physical_cup_freejoint=True,physics_stepped=True,
                max_cup_rise_m=max(r['cup_rise_m'] for r in records),
                final_cup_rise_m=records[-1]['cup_rise_m'],
                final_wrist_lift_world_z_m=records[-1]['wrist_lift_world_z_m'],
                hold_min_rise_m=min((r['cup_rise_m'] for r in hold),default=0),
                hold_max_finger_contacts=max((r['finger_contacts'] for r in hold),default=0),
                hold_max_thumb_contacts=max((r['thumb_contacts'] for r in hold),default=0),
                maximum_penetration_m=max(r['max_cup_penetration_m'] for r in records),
                max_cup_xy_shift_m=max(r['cup_xy_shift_m'] for r in records),
                ik_failed_targets=failures,
                success=(bool(hold) and min(r['cup_rise_m'] for r in hold)>.03
                         and any(r['finger_contacts']>=2 for r in hold)
                         and any(r['thumb_contacts']>=1 for r in hold)),
                note='Success is preliminary. Inspect penetrations, position mapping, and replay video.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)




def _dex5_draw_palm_marker(model, data, handle, offset_mm):
    """Draw a palm-attached RGB coordinate frame in the MuJoCo viewer."""
    palm_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY,
        "g1_01_dex5_right_palm"
    )
    if palm_id < 0:
        raise RuntimeError("DEX5 palm body not found")

    R = data.xmat[palm_id].reshape(3, 3)

    # Approximate geometric centre of the palm body.
    geom_ids = np.flatnonzero(model.geom_bodyid == palm_id)

    if len(geom_ids):
        centre = data.geom_xpos[geom_ids].mean(axis=0).copy()
    else:
        centre = data.xpos[palm_id].copy()

    # Optional adjustment expressed in palm-local coordinates.
    centre += R @ (np.asarray(offset_mm, dtype=float) / 1000.0)

    # Rotate marker +30 degrees around its local red (X) axis.
    angle = np.deg2rad(-30.0)
    c, sn = np.cos(angle), np.sin(angle)

    Rx = np.array([
        [1, 0, 0],
        [0, c, -sn],
        [0, sn, c]
    ])

    R_marker = R @ Rx

    colours = [
        (1.0, 0.1, 0.1, 1.0),  # X - red
        (0.1, 1.0, 0.1, 1.0),  # Y - green
        (0.1, 0.4, 1.0, 1.0),  # Z - blue
    ]

    with handle.lock():
        scene = handle.user_scn
        scene.ngeom = 0

        # Palm centre marker.
        mujoco.mjv_initGeom(
            scene.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.008, 0.008, 0.008]),
            centre,
            np.eye(3).flatten(),
            np.array([1.0, 1.0, 0.0, 1.0])
        )
        scene.ngeom = 1

        # Orientation axes, 60 mm each.
        for axis in range(3):
            geom = scene.geoms[scene.ngeom]
            endpoint = centre + R_marker[:, axis] * 0.060

            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.003,
                centre,
                endpoint
            )
            geom.rgba[:] = colours[axis]
            scene.ngeom += 1



def _dex5_palm_pose(model, data, offset_mm):
    pid = model.body("g1_01_dex5_right_palm").id
    R = data.xmat[pid].reshape(3, 3)

    ids = np.flatnonzero(model.geom_bodyid == pid)
    centre = (
        data.geom_xpos[ids].mean(axis=0).copy()
        if len(ids) else data.xpos[pid].copy()
    )

    centre += R @ (np.asarray(offset_mm, float) / 1000.)

    a = np.deg2rad(-30.0)
    c, si = np.cos(a), np.sin(a)

    Rx = np.array([
        [1., 0., 0.],
        [0., c, -si],
        [0., si, c]
    ])

    return centre, R @ Rx


def view(args):
    from mujoco import viewer
    import cv2

    scene = args.output / "scenario_dynamic_replay.xml"
    replay = args.output / "physics_replay.npz"

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    with np.load(replay) as saved:
        qs = saved["qpos"].copy()
        times = saved["time_s"].copy()

    data.qpos[:] = qs[0]
    mujoco.mj_forward(model, data)

    cap = None
    if args.video_view:
        root = args.root.expanduser().resolve()
        camera_file = (
            root / "code/stereo/recordings_three_cam"
            / args.take / "camera1.avi"
        )
        trajectory_file = (
            args.trajectory.expanduser().resolve() if args.trajectory else
            root / "outputs" / f"smoothed_{args.take}" / "sim_qpos.npz"
        )

        with np.load(trajectory_file) as source:
            recorded_times = source["time_s"].copy()
            recorded_frames = source["source_frames"].copy()

        recorded_times -= recorded_times[0]

        matches = np.where(recorded_frames == args.grasp_frame)[0]
        if len(matches) != 1:
            raise RuntimeError("Grasp frame not found")

        grasp_index = int(matches[0])
        grasp_time = float(recorded_times[grasp_index])

        cap = cv2.VideoCapture(str(camera_file))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {camera_file}")

        cv2.namedWindow("Original Camera - Synced", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original Camera - Synced", 960, 540)
        cv2.moveWindow("Original Camera - Synced", 30, 100)

    handle = viewer.launch_passive(model, data)

    cup_id = model.body("table_cup").id
    handle.cam.lookat[:] = data.xpos[cup_id] + [0, 0, 0.12]
    handle.cam.distance = 0.85
    handle.cam.azimuth = 130
    handle.cam.elevation = -20

    start_time = time.monotonic()
    paused = False
    pause_time = 0.0
    last_frame = -1

    try:
        while handle.is_running():
            elapsed = (
                pause_time if paused
                else time.monotonic() - start_time
            )

            if elapsed > times[-1]:
                if args.close_after:
                    break
                start_time = time.monotonic()
                elapsed = 0.0
                pause_time = 0.0
                last_frame = -1

            index = int(np.clip(
                np.searchsorted(times, elapsed),
                0, len(qs) - 1
            ))

            with handle.lock():
                data.qpos[:] = qs[index]
                mujoco.mj_forward(model, data)

            _dex5_draw_palm_marker(model, data, handle, args.palm_marker_offset_mm)
            handle.sync()

            if cap is not None:
                # The original recording ends at grasp.
                camera_time = np.clip(elapsed - 0.4, 0, grasp_time)

                ri = int(np.clip(
                    np.searchsorted(recorded_times, camera_time),
                    0, grasp_index
                ))
                frame_id = int(recorded_frames[ri])

                if frame_id != last_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ok, frame = cap.read()

                    if not ok:
                        raise RuntimeError(
                            f"Cannot read camera frame {frame_id}"
                        )

                    frame = cv2.resize(frame, (960, 540))

                    cv2.putText(
                        frame,
                        f"Frame {frame_id} | Simulation {elapsed:.2f}s",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2
                    )

                    if elapsed > grasp_time + 0.4:
                        cv2.putText(
                            frame,
                            "CAMERA HELD AT GRASP",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 255), 2
                        )

                    cv2.imshow("Original Camera - Synced", frame)
                    last_frame = frame_id

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == 32:
                    paused = not paused
                    if paused:
                        pause_time = elapsed
                    else:
                        start_time = time.monotonic() - pause_time

                if key == ord("r"):
                    start_time = time.monotonic()
                    pause_time = 0.0
                    paused = False
                    last_frame = -1

            time.sleep(0.01)

    finally:
        handle.close()
        if cap is not None:
            cap.release()
            cv2.destroyAllWindows()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.home()/'physikal_workflow/dex-5')
    ap.add_argument('--take',default='take_20261104_175536_951802')
    ap.add_argument('--trajectory', type=Path, help='Explicit input sim_qpos.npz from kinematic replay')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--grasp-frame',type=int,default=85)
    ap.add_argument('--mass-kg',type=float,default=.15)
    ap.add_argument('--friction',type=float,default=1.)
    ap.add_argument('--finger-torque',type=float,default=.93)
    ap.add_argument('--timestep',type=float,default=.001)
    ap.add_argument('--lift-height',type=float,default=.06)
    ap.add_argument('--cup-at-palm',action='store_true')
    ap.add_argument('--palm-distance',type=float,default=0.075)
    ap.add_argument('--palm-sideways',type=float,default=0.0)
    ap.add_argument('--fixed-cup-table',action='store_true')
    ap.add_argument('--cup-table-x',type=float,default=0.188335)
    ap.add_argument('--cup-table-y',type=float,default=-0.047794)
    ap.add_argument(
        '--palm-marker-offset-mm',
        nargs=3,
        type=float,
        default=[0, 0, 0],
        metavar=('X', 'Y', 'Z')
    )
    ap.add_argument('--cup-from-approach-blue',action='store_true')
    ap.add_argument('--cup-clearance-mm',type=float,default=20.)
    ap.add_argument('--view',action='store_true')
    ap.add_argument('--video-view',action='store_true')
    ap.add_argument('--close-after',action='store_true')
    a=ap.parse_args();a.output=a.output.expanduser().resolve()
    if a.view:view(a)
    else:run(a)


if __name__=='__main__':main()
