#!/usr/bin/env python3
"""DEX-5 video -> G1 right-hand kinematic MuJoCo preview.

Inputs:
 * FoundationStereo/wristband_trial_new/palm_pose_6dof.csv
   SAM3 wristband 3-D surface points and human anatomical palm quaternions.
 * code/../outputs/.../joint_targets.csv from existing 10_g1_dex5_stationary.py
   verified original Dex5 per-finger joint names and thumb fit.
 * models/scenario_g1_dex5_stationary.xml

Transforms table->MuJoCo by a fixed yaw and a diagnostic position translation
anchored to the initial robot hand site unless a table origin is supplied.
XY-only registration is supported and keeps the earlier Z anchor. Anatomical hand orientation does not use a first-frame reference.
No physics stepping, grasp/object motion, contact validation, collision checking,
or walking. Invalid palm orientations are held, NOT fabricated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

ARM = [
    'g1_01_right_shoulder_pitch_joint',
    'g1_01_right_shoulder_roll_joint',
    'g1_01_right_shoulder_yaw_joint',
    'g1_01_right_elbow_joint',
    'g1_01_right_wrist_roll_joint',
    'g1_01_right_wrist_pitch_joint',
    'g1_01_right_wrist_yaw_joint',
]
SITES = {'dex5': 'g1_01_dex5_right_frame', 'wrist': 'g1_01_right_wrist_ik_site'}
FINGER_ANCHORS = ('Roll_21R', 'Roll_31R', 'Roll_51R')


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def xyz_from(row, names):
    try:
        v = np.array([float(row[k]) for k in names], dtype=float)
        return v if np.isfinite(v).all() else None
    except (KeyError, ValueError, TypeError):
        return None


def normalize(x):
    x = np.asarray(x, float)
    size = np.linalg.norm(x)
    if not np.isfinite(size) or size < 1e-9:
        raise ValueError('Degenerate hand geometry')
    return x / size


def palm_basis(wrist, index, middle, pinky):
    y = normalize(middle-wrist)
    x0 = index-pinky
    x = normalize(x0-y*np.dot(x0, y))
    z = normalize(np.cross(x,y))
    return np.column_stack((x,y,z))


def get_joint(mujoco, model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError('Missing joint in MuJoCo model: ' + name)
    return int(jid), int(model.jnt_qposadr[jid])


def get_orientation(row):
    if row.get('palm_status') != 'VALID':
        return None
    q = xyz_from(row, ('palm_qx_table','palm_qy_table','palm_qz_table','palm_qw_table'))
    if q is None or abs(np.linalg.norm(q)-1.) > 0.03:
        return None
    return Rotation.from_quat(q/np.linalg.norm(q)).as_matrix()


def trajectory_segments(samples, targets, fps):
    """Connect measured contiguous samples only, never across missing XYZ."""
    segments = []
    for previous, current, start, end in zip(samples, samples[1:], targets, targets[1:]):
        dt = current[2]-previous[2]
        if (current[0] != previous[0]+1 or not np.isfinite(dt)
                or dt <= 0 or dt > max(.4, 4/fps)):
            continue
        if np.isfinite(start).all() and np.isfinite(end).all():
            if np.linalg.norm(np.asarray(end)-start) > 1e-8:
                segments.append((start, end))
    return segments


def apply_finger_targets(qpos, row, bindings):
    """Apply existing retargeted angles; missing values hold that joint."""
    applied = 0
    for key, (address, limits) in bindings.items():
        try:
            angle = float(row.get(key, ''))
        except (TypeError, ValueError):
            continue
        if np.isfinite(angle):
            qpos[address] = np.clip(angle, *limits)
            applied += 1
    return applied


def complete_frame_timeline(selected, finger_lookup, bindings, arm_addresses,
                            rest_qpos, rest_position, solved_qpos, solved_trace,
                            solved_targets, solved_rotations):
    """Merge existing IK solutions and original finger targets by source frame.

    Missing XYZ holds the arm without dropping RGB or finger frames. No new
    finger measurement or arm target is inferred for those frames.
    """
    solved = {int(row['source_video_frame']): i for i, row in enumerate(solved_trace)}
    if len(solved) != len(solved_trace):
        raise ValueError('Duplicate solved source frames')
    if any(b[1] <= a[1] or b[2] <= a[2] for a, b in zip(selected, selected[1:])):
        raise ValueError('Source frames/timestamps must increase')
    qpos = rest_qpos.copy()
    actual = np.asarray(rest_position).copy()
    qs, times, targets, rotations, trace, source_frames = [], [], [], [], [], []
    origin = selected[0][2]
    for seq, frame, timestamp, row in selected:
        index = solved.get(frame)
        if index is not None:
            qpos[arm_addresses] = solved_qpos[index][arm_addresses]
            record = dict(solved_trace[index])
            actual = np.array([record['wrist_actual_'+axis] for axis in 'xyz'])
            target = solved_targets[index].copy()
            rotation = solved_rotations[index].copy()
        else:
            record = dict(solved_trace[0])
            record.update(pose_source=row.get('palm_status', 'INVALID'),
                          orientation_source='HOLD_INVALID_XYZ', orientation_ik_active=False,
                          raw_target_xyz_m=[float('nan')]*3, filtered_target_delta_mm=float('nan'),
                          ik_status='HOLD_INVALID_XYZ', position_error_mm=float('nan'),
                          orientation_error_deg=float('nan'))
            target = np.full(3, np.nan)
            rotation = np.eye(3)
            for axis, value in zip('xyz', actual):
                record['wrist_actual_'+axis] = float(value)
                record['wrist_target_'+axis] = float('nan')
        finger = finger_lookup.get(frame, {})
        applied = apply_finger_targets(qpos, finger, bindings)
        record.update(sequence_frame=seq, source_video_frame=frame,
                      timestamp_s=timestamp-origin, wrist_xyz_valid=index is not None,
                      finger_joints_applied=applied, thumb_state=finger.get('thumb_state','NO_FINGER'))
        qs.append(qpos.copy()); times.append(timestamp-origin); targets.append(target)
        rotations.append(rotation); trace.append(record); source_frames.append(frame)
    return qs, times, targets, rotations, trace, source_frames


def add_pose_markers(mujoco, viewer, target_position, target_rotation,
                     actual_position, actual_rotation, path_segments=()):
    """Show target hand-site pose and actual IK hand-site pose in MuJoCo world.

    YELLOW sphere = target site (currently mapped from wristband surface).
    CYAN sphere = actual selected IK site used by IK.
    MAGENTA line = XYZ residual from target to actual.
    Bright RGB arrows = target site axes; dim RGB arrows = actual site axes.
    BLUE cylinders = complete smoothed target path, broken at missing XYZ.
    """
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        def sphere(p, rgba, radius):
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius), p,
                np.eye(3).ravel(), np.asarray(rgba, np.float32))
            scene.ngeom += 1
        def segment(p0, p1, rgba, radius=.003, kind=None):
            if scene.ngeom >= scene.maxgeom:
                return
            kind = mujoco.mjtGeom.mjGEOM_LINE if kind is None else kind
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, kind, np.full(3, radius),
                np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
            mujoco.mjv_connector(geom, kind, radius,
                np.asarray(p0, float), np.asarray(p1, float))
            scene.ngeom += 1
        target_valid = np.isfinite(target_position).all()
        if target_valid:
            sphere(target_position, (1.0, .75, .08, 1.0), .017)
        sphere(actual_position, (.05, .9, 1.0, 1.0), .014)
        if target_valid:
            segment(target_position, actual_position, (1.0, 0.0, .85, 1.0), .004,
                    mujoco.mjtGeom.mjGEOM_LINE)
        for origin, rotation, dim in ((target_position, target_rotation, 1.0),
                                      (actual_position, actual_rotation, .45)):
            if not np.isfinite(origin).all():
                continue
            for axis, color in enumerate(((1.,0.,0.), (0.,1.,0.), (0.,0.,1.))):
                endpoint = origin + rotation[:, axis] * (.075 if dim == 1.0 else .055)
                segment(origin, endpoint, (*[v * dim for v in color], 1.), .004,
                        mujoco.mjtGeom.mjGEOM_ARROW)
        for start, end in path_segments:
            if scene.ngeom >= scene.maxgeom:
                break
            segment(start, end, (.12, .4, 1., 1.), .0025,
                    mujoco.mjtGeom.mjGEOM_CYLINDER)



def filter_targets(samples, fps, gap=0.30, xyz_win=9, rot_win=7):
    n = len(samples)
    raw = np.array([r[3] for r in samples], float)
    xyz = raw.copy()
    rot = [None if r[4] is None else r[4].copy() for r in samples]
    flags = ['MEASURED' if r is not None else 'INVALID' for r in rot]

    blocks = []
    lo = 0
    for i in range(1, n):
        ta, tb = samples[i-1][2], samples[i][2]
        dt = tb-ta if np.isfinite(ta) and np.isfinite(tb) else 1/fps
        if samples[i][0] != samples[i-1][0]+1 or dt > max(0.4, 4/fps):
            blocks.append((lo, i))
            lo = i
    blocks.append((lo, n))

    for lo, hi in blocks:
        k = hi-lo
        if k >= 3:
            med = median_filter(raw[lo:hi], size=(3,1), mode='nearest')
            clean = raw[lo:hi].copy()
            bad = np.linalg.norm(clean-med, axis=1) > .025
            clean[bad] = med[bad]
            win = min(xyz_win, k if k%2 else k-1)
            xyz[lo:hi] = (
                savgol_filter(clean, win, 2 if win>=5 else 1,
                              axis=0, mode='interp')
                if win>=3 else clean
            )

        # Reject isolated orientation flips.
        for i in range(lo+1, hi-1):
            if any(rot[j] is None for j in (i-1,i,i+1)):
                continue
            angle = lambda a,b: np.degrees(
                Rotation.from_matrix(a.T@b).magnitude()
            )
            if (angle(rot[i-1],rot[i])>45
                    and angle(rot[i],rot[i+1])>45
                    and angle(rot[i-1],rot[i+1])<25):
                rot[i] = None
                flags[i] = 'REJECTED_SPIKE'

        valid = [i for i in range(lo,hi) if rot[i] is not None]

        # Interpolate only short missing orientation intervals.
        for a,b in zip(valid,valid[1:]):
            if b-a<=1:
                continue
            t0,t1 = samples[a][2],samples[b][2]
            duration = (
                t1-t0 if np.isfinite(t0) and np.isfinite(t1) and t1>t0
                else (b-a)/fps
            )
            if duration>gap:
                continue

            interp = Slerp(
                [0,1],
                Rotation.from_matrix([rot[a],rot[b]])
            )
            for i in range(a+1,b):
                ti=samples[i][2]
                fraction=(ti-t0)/(t1-t0) if np.isfinite(ti) and np.isfinite(t0) and np.isfinite(t1) and t1>t0 else (i-a)/(b-a)
                rot[i] = interp([np.clip(fraction,0,1)]).as_matrix()[0]
                flags[i] = 'SHORT_GAP_INTERPOLATED'

        # Rotation-space smoothing.
        orig = [None if r is None else r.copy() for r in rot]
        radius = rot_win//2
        sigma = max(1., rot_win/4.)

        for i in range(lo,hi):
            center = orig[i]
            if center is None:
                continue
            vecs,weights = [],[]
            for j in range(max(lo,i-radius),min(hi,i+radius+1)):
                if orig[j] is None:
                    continue
                vecs.append(
                    Rotation.from_matrix(
                        center.T@orig[j]
                    ).as_rotvec()
                )
                weights.append(
                    np.exp(-.5*((i-j)/sigma)**2)
                )
            rot[i] = center @ Rotation.from_rotvec(
                np.average(vecs,axis=0,weights=weights)
            ).as_matrix()

    return xyz,rot,flags

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path.home()/'physikal_workflow/dex-5'
    ap.add_argument('--palm-csv', type=Path, default=root/'FoundationStereo/wristband_trial_new/palm_pose_6dof.csv')
    ap.add_argument('--finger-csv', type=Path, required=True, help='joint_targets.csv generated by 10_g1_dex5_stationary.py --no-wrist')
    ap.add_argument('--model', type=Path, default=root/'models/scenario_g1_dex5_stationary.xml')
    ap.add_argument('--output', type=Path, default=root/'outputs/dex5_video_to_sim_v1')
    ap.add_argument('--ik-reference', choices=('dex5','wrist'), default='wrist',
                    help='Select IK site: original Dex5 palm mount or new wrist-yaw joint-centred site')
    ap.add_argument('--ik-mode', choices=('xyz','pose'), default='xyz',
                    help='xyz: position-first diagnostic; pose: try position + absolute palm orientation')
    ap.add_argument('--table-yaw-deg', type=float, default=-90.,
                    help='Verified table frame -> MuJoCo world yaw for this setup')
    ap.add_argument('--table-origin-world', type=float, nargs=3, metavar=('X','Y','Z'),
                    help='Absolute MuJoCo world position of calibrated table origin (omit for diagnostic initial XYZ anchoring)')
    ap.add_argument('--table-origin-xy-world', type=float, nargs=2, metavar=('X','Y'),
                    help='MuJoCo XY of physical table calibration origin; replaces first-frame anchor ONLY in XY. Z remains anchored.')
    ap.add_argument('--robot-alpha', type=float, default=0.32,
                    help='Opacity of G1/Dex5 visual geoms in viewer (0.05..1.0); table remains opaque')
    ap.add_argument('--pose-weight', type=float, default=.5, help='Orientation residual weight (pose IK only)')
    ap.add_argument('--position-weight', type=float, default=10., help='Position residual weight')
    ap.add_argument('--max-arm-speed', type=float, default=4., help='Per-joint speed cap, rad/s')
    ap.add_argument('--xyz-window',type=int,default=9)
    ap.add_argument('--orientation-window',type=int,default=7)
    ap.add_argument('--orientation-max-gap-s',type=float,default=.30)
    ap.add_argument('--fps', type=float, default=15., help='Playback FPS fallback if capture timestamps missing')
    ap.add_argument('--speed', type=float, default=.5, help='Visualization playback speed factor')
    ap.add_argument('--start', type=int, default=0, help='Prepared trial sequence index, inclusive')
    ap.add_argument('--end', type=int, help='Prepared trial sequence index, inclusive')
    ap.add_argument('--headless', action='store_true', help='Save trajectory and trace without opening viewer')
    ap.add_argument('--video', type=Path, help='Source camera1.avi to show beside MuJoCo (auto-detected from trial meta.json if omitted)')
    ap.add_argument('--video-width', type=int, default=720, help='Video window width in pixels')
    ap.add_argument('--autoplay', action='store_true', help='Start playing immediately; default starts paused')
    ap.add_argument('--preposition-seconds', type=float, default=2.0,
                    help='Time to move from symmetric resting pose to first solved IK frame (seconds)')
    a=ap.parse_args()
    if a.max_arm_speed<=0 or a.fps<=0 or a.position_weight<=0 or not 0<a.speed<=4 or a.preposition_seconds<0:
        ap.error('Invalid speed, FPS or position weight')
    if a.table_origin_world is not None and a.table_origin_xy_world is not None:
        ap.error('Use either --table-origin-xy-world or --table-origin-world, not both')
    if a.table_origin_xy_world is not None and not np.isfinite(a.table_origin_xy_world).all():
        ap.error('--table-origin-xy-world must contain finite coordinates')
    if not .05 <= a.robot_alpha <= 1.:
        ap.error('--robot-alpha must be between 0.05 and 1.0')
    import mujoco
    palm_rows = read_csv(a.palm_csv.expanduser())
    finger_rows = read_csv(a.finger_csv.expanduser())
    if not palm_rows or not finger_rows:
        ap.error('Empty palm or Dex5 finger CSV')
    finger_lookup = {}
    for r in finger_rows:
        try: finger_lookup[int(r['frame'])] = r
        except (KeyError, ValueError): pass
    model=mujoco.MjModel.from_xml_path(str(a.model.expanduser().resolve()))
    data=mujoco.MjData(model)
    data.qpos[:] = model.qpos0.copy()
    for name,val in (('g1_01_right_shoulder_roll_joint',-.25),('g1_01_right_elbow_joint',.6)):
        j,adr=get_joint(mujoco,model,name)
        lo,hi=model.jnt_range[j]
        data.qpos[adr]=np.clip(val,lo,hi)
    mujoco.mj_forward(model,data)
    site_name=SITES[a.ik_reference]
    sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,site_name)
    if sid<0: ap.error('IK site missing from loaded MuJoCo model: '+site_name+'; ensure the scenario includes the edited robot XML')
    palm_sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,SITES['dex5'])
    if palm_sid<0: ap.error('Original Dex5 site missing: '+SITES['dex5'])
    arm_ids,arm_addrs=zip(*(get_joint(mujoco,model,n) for n in ARM))
    arm_addrs=np.asarray(arm_addrs,int)
    limits=np.asarray(model.jnt_range[list(arm_ids)],float)
    if np.any(limits[:,1] <= limits[:,0]):
        raise ValueError('Joint limits missing or invalid')
    qcur=data.qpos[arm_addrs].copy()
    initial_hand_pos=data.site_xpos[sid].copy()
    initial_site_rot=data.site_xmat[sid].reshape(3,3).copy()
    # Display/reset both arms at model rest, not the right-only IK seed above.
    rest_qpos=model.qpos0.copy()
    janchors=[get_joint(mujoco,model,'g1_01_dex5_right_'+n)[0] for n in FINGER_ANCHORS]
    # Always construct the anatomical basis from the SAME physical Dex5 palm
    # site in both IK modes. Only the solved end-effector reference changes.
    dex_anat=palm_basis(data.site_xpos[palm_sid].copy(),*[data.xanchor[j].copy() for j in janchors])
    site_to_anat=initial_site_rot.T@dex_anat
    if not np.allclose(site_to_anat.T@site_to_anat,np.eye(3),atol=1e-4):
        raise RuntimeError('Bad Dex5 site-to-anatomical transform')
    yaw=Rotation.from_euler('z',a.table_yaw_deg,degrees=True).as_matrix()
    usable=[]
    selected_rows=[]
    for row in palm_rows:
        try: seq=int(row['sequence_frame']);frame=int(row['source_video_frame'])
        except (KeyError, ValueError): continue
        if seq<a.start or (a.end is not None and seq>a.end):continue
        try: t=float(row.get('timestamp_s','nan'))
        except ValueError: t=float('nan')
        if not np.isfinite(t):t=seq/a.fps
        selected_rows.append((seq,frame,t,row))
        xyz=xyz_from(row,('X_table_m','Y_table_m','Z_table_m'))
        if xyz is None:continue
        q=get_orientation(row)
        usable.append((seq,frame,t,xyz,q,row.get('palm_status','?')))
    if not usable:raise RuntimeError('No valid tabletop XYZ in selected palm CSV rows')
    smoothed_xyz,smoothed_rot,ori_flags=filter_targets(usable,a.fps,a.orientation_max_gap_s,a.xyz_window,a.orientation_window)
    print('SMOOTHED wrist target; XYZ window:',a.xyz_window,'rotation window:',a.orientation_window,flush=True)
    # Keep first-frame Z in XY-only testing; existing intentional 70 mm
    # workspace lift is preserved by this anchored Z convention.
    anchor_offset=initial_hand_pos - yaw@usable[0][3]
    if a.table_origin_xy_world is not None:
        offset=anchor_offset.copy()
        offset[:2]=np.asarray(a.table_origin_xy_world,float)
        position_mapping='TABLE-ORIGIN XY; DIAGNOSTIC FIRST-FRAME Z (XY alignment hypothesis until origin verified)'
    elif a.table_origin_world is None:
        offset=anchor_offset.copy()
        position_mapping='DIAGNOSTIC first-position anchor (not calibrated cup/table world)'
    else:
        offset=np.asarray(a.table_origin_world,float)
        position_mapping='CALIBRATED-INPUT table origin (provided by user; verify independently)'
    out=a.output.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    bindings={}
    for key in finger_rows[0].keys():
        if key.startswith(('Yaw_','Pitch_','Roll_')):
            name='g1_01_dex5_right_'+key
        elif key.startswith('g1_01_dex5_right_'):
            name=key
        else:
            continue
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if jid<0:
            raise ValueError('Retargeted finger joint is missing from MuJoCo model: '+name)
        if jid>=0:
            bindings[key]=(int(model.jnt_qposadr[jid]), np.asarray(model.jnt_range[jid],float))
    if len(bindings)<8:
        raise ValueError(f'Only {len(bindings)} Dex5 joints matched finger CSV! Recheck file or model. CSV columns: {list(finger_rows[0])}')
    print(f'Palm frames: {len(usable)}; Dex5 finger bindings: {len(bindings)}',flush=True)
    print(f'Position map: {position_mapping}; offset={offset.round(4)}',flush=True)
    print(f'Orientation: absolute anatomical palm -> Dex5 site; Rz(table yaw)={a.table_yaw_deg} degrees; no first-frame rotation calibration',flush=True)
    print('This is KINEMATIC ONLY: NO robot physics, collision avoidance, object grasp, or contact validation.',flush=True)
    print('Missing/rejected palm orientations HOLD the last valid rotation. Wristband is visible SURFACE, not wrist joint centre.',flush=True)
    print(f'IK mode: {a.ik_mode}; reference: {a.ik_reference} ({site_name})',flush=True)
    trace=[];qs=[];times=[];targets=[];target_rotations=[];orientation_valid=0
    last_orientation=None
    last_finger=None
    initial_time=None
    previous_t=None
    for k,(seq,frame,t,xyz_table,r_palm,status) in enumerate(usable):
        if not np.isfinite(t):t=seq/a.fps
        if initial_time is None: initial_time=t
        t=float(t-initial_time)
        if previous_t is None: dt=1./a.fps
        else:dt=max(1./a.fps*.5, min(.2,t-previous_t))
        previous_t=t
        target_pos=yaw@smoothed_xyz[k] + offset
        r_palm=smoothed_rot[k]
        if r_palm is not None:
            orientation_valid+=1
            last_orientation=yaw@r_palm @ site_to_anat.T
            ori_source=ori_flags[k]
        elif last_orientation is not None:
            ori_source='HOLD_INVALID'
        else:
            last_orientation=initial_site_rot.copy()
            ori_source='UNINITIALIZED_HOLD'
        target_rot=last_orientation.copy()
        previous=qcur.copy()
        def fk(q):
            data.qpos[arm_addrs]=q
            mujoco.mj_forward(model,data)
            return data.site_xpos[sid].copy(),data.site_xmat[sid].reshape(3,3).copy()
        def residual(q):
            p,r=fk(q)
            parts=[a.position_weight*(p-target_pos)]
            if a.ik_mode=='pose' and r_palm is not None:
                parts.append(a.pose_weight*Rotation.from_matrix(target_rot@r.T).as_rotvec())
            parts.append(.025*(q-previous))
            return np.concatenate(parts)
        fit=least_squares(residual, previous, bounds=(limits[:,0]+1e-7,limits[:,1]-1e-7),
                          max_nfev=75,xtol=1e-5,ftol=1e-5,gtol=1e-5)
        # Cap joint motion per source interval so solver cannot teleport joints.
        change=np.clip(fit.x-previous,-a.max_arm_speed*dt,a.max_arm_speed*dt)
        qcur=np.clip(previous+change,limits[:,0],limits[:,1]);p,r=fk(qcur)
        pos_err=float(np.linalg.norm(p-target_pos)*1000)
        ori_err=float(np.degrees(Rotation.from_matrix(target_rot@r.T).magnitude()))
        if pos_err<=25 and (a.ik_mode=='xyz' or r_palm is None or ori_err<=25):ik_status='TRACKING'
        elif pos_err<=70:ik_status='APPROX'
        else:ik_status='LARGE_ERROR'
        finger=finger_lookup.get(frame)
        if finger is not None:
            last_finger=finger
        applied=0
        if last_finger is not None:
            for key,(adr,bnd) in bindings.items():
                try: angle=float(last_finger.get(key,''))
                except (ValueError,TypeError):continue
                if np.isfinite(angle):
                    data.qpos[adr]=np.clip(angle,*bnd)
                    applied+=1
        mujoco.mj_forward(model,data)
        data.time=t
        qs.append(data.qpos.copy())
        times.append(t)
        targets.append(target_pos.copy())
        target_rotations.append(target_rot.copy())
        trace.append(dict(sequence_frame=seq,source_video_frame=frame,timestamp_s=t,
                          pose_source=status,orientation_source=ori_source,
                          orientation_ik_active=(a.ik_mode=='pose' and r_palm is not None),
                          raw_target_xyz_m=(yaw@xyz_table+offset).tolist(),
                          filtered_target_delta_mm=float(np.linalg.norm(smoothed_xyz[k]-xyz_table)*1000),
                          ik_status=ik_status,
                          position_error_mm=round(pos_err,3),orientation_error_deg=round(ori_err,3),
                          wrist_target_x=target_pos[0],wrist_target_y=target_pos[1],wrist_target_z=target_pos[2],
                          wrist_actual_x=p[0],wrist_actual_y=p[1],wrist_actual_z=p[2],
                          finger_joints_applied=applied,thumb_state=last_finger.get('thumb_state','?') if last_finger else 'NO_FINGER'))
        if seq%10==0 or seq==usable[-1][0]:
            print(f'{seq+1:3d}: ARM={ik_status:11} pos={pos_err:5.1f}mm rot={ori_err:5.1f}deg palm={ori_source:12} finger={applied}',flush=True)
    path_segments = trajectory_segments(usable, targets, a.fps)
    rest_data=mujoco.MjData(model);rest_data.qpos[:]=rest_qpos;mujoco.mj_forward(model,rest_data)
    qs,times,targets,target_rotations,trace,source_frames=complete_frame_timeline(
        selected_rows,finger_lookup,bindings,arm_addrs,rest_qpos,rest_data.site_xpos[sid],
        qs,trace,targets,target_rotations)
    np.savez_compressed(out/'sim_qpos.npz',qpos=np.asarray(qs),time_s=np.asarray(times),
                        source_frames=np.asarray(source_frames),target_pos=np.asarray(targets),
                        wrist_xyz_valid=np.asarray([row['wrist_xyz_valid']for row in trace]),
                        target_quat_xyzw=Rotation.from_matrix(np.asarray(target_rotations)).as_quat())
    with (out/'sim_quality.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(trace[0].keys()));w.writeheader();w.writerows(trace)
    summary=dict(frames=len(qs),measured_palm_orientations=orientation_valid,
                 wrist_xyz_valid_frames=len(usable),held_wrist_frames=len(qs)-len(usable),
                 finger_bindings={key:address for key,(address,_)in bindings.items()},
                 finger_updates_during_missing_xyz=sum(not v['wrist_xyz_valid'] and v['finger_joints_applied']>0 for v in trace),
                 with_finger_targets=sum(v['finger_joints_applied']>=8 for v in trace),
                 position_mapping=position_mapping,table_origin_to_world_translation=offset.tolist(),
                 ik_mode=a.ik_mode,ik_reference=a.ik_reference,ik_site=site_name,statuses={s:sum(v['ik_status']==s for v in trace) for s in ('TRACKING','APPROX','LARGE_ERROR','HOLD_INVALID_XYZ')},
                 median_position_error_mm=float(np.nanmedian([v['position_error_mm'] for v in trace])),
                 median_orientation_error_deg=float(np.nanmedian([v['orientation_error_deg'] for v in trace])),
                 physics=False,collision_checked=False,grasp_tested=False)
    (out/'sim_report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Simulation trace:',out/'sim_quality.csv',flush=True)
    print('Qpos trajectory:',out/'sim_qpos.npz',flush=True)
    print('Report:',out/'sim_report.json',flush=True)
    print('Statuses:',summary['statuses'],flush=True)
    if a.headless:return
    # Visualization only; no changes to collision or solver geometry.
    num_translucent = 0
    if a.robot_alpha < 1.0:
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            robot_ancestor = False
            while body_id > 0:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ''
                if body_name.startswith('g1_01') or 'dex5' in body_name.lower():
                    robot_ancestor = True
                    break
                body_id = int(model.body_parentid[body_id])
            if robot_ancestor:
                model.geom_rgba[geom_id, 3] = min(model.geom_rgba[geom_id, 3], a.robot_alpha)
                num_translucent += 1
    print(f'Robot transparency: modified {num_translucent} G1/Dex5 geoms; alpha={a.robot_alpha}', flush=True)
    import cv2
    from mujoco import viewer as mjv

    # Video and robot replay use the same indexed source frames, not an
    # independent video-player clock. This keeps both streams synchronized.
    video_path = a.video.expanduser() if a.video else None
    if video_path is None:
        trial_meta = a.palm_csv.expanduser().parent / 'meta.json'
        if trial_meta.is_file():
            meta = json.loads(trial_meta.read_text())
            source_take = meta.get('take')
            if source_take:
                video_path = Path(source_take).expanduser() / 'camera1.avi'
    if video_path is None or not video_path.is_file():
        raise FileNotFoundError(
            'Source video not found. Supply --video /path/to/take_.../camera1.avi '
            '(or ensure wristband_trial_new/meta.json contains its source take).'
        )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    if a.video_width < 240 or a.video_width > 1920:
        raise ValueError('--video-width must be 240..1920')
    window = 'DEX-5 human video [P=play/pause, R=restart, Q=quit]'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cap_frame_index = -1

    def get_source_frame(idx):
        nonlocal cap_frame_index
        wanted = int(source_frames[idx])
        if wanted != cap_frame_index + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, wanted)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Failed to read source frame {wanted} of {video_path}')
        cap_frame_index = wanted
        return frame

    def show_video(idx, paused):
        frame = get_source_frame(idx)
        h, w = frame.shape[:2]
        vis = cv2.resize(frame, (a.video_width, int(h*a.video_width/w)))
        status = trace[idx]
        # Do not rotate Camera 1. Camera 2's 180-degree rotation is irrelevant.
        lines = [
            f"{'PAUSED' if paused else 'PLAYING'}  [{idx+1}/{len(qs)}]  source frame {source_frames[idx]}",
            f"IK {status['ik_status']}  pos {status['position_error_mm']:.1f}mm  rot {status['orientation_error_deg']:.1f}deg",
            f"Palm {status['orientation_source']}   P: play/pause   R: restart   Q: quit",
            f"MuJoCo: YELLOW=target  CYAN=actual {a.ik_reference} IK site  PINK=error",
        ]
        for j, line in enumerate(lines):
            y = 28 + j*27
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, vis)

    data.qpos[:]=rest_qpos
    mujoco.mj_forward(model,data)
    viewer=mjv.launch_passive(model,data)
    viewer.cam.lookat[:]=data.site_xpos[sid]
    viewer.cam.distance=1.5
    viewer.cam.azimuth=140
    viewer.cam.elevation=-18
    playing = bool(a.autoplay)
    i = 0
    last_shown = -1
    phase = 'preposition' if playing else 'rest'
    transition_start = time.monotonic() if playing else None
    next_advance = time.monotonic()
    print(f'Human video: {video_path}',flush=True)
    print('Playback controls (focus VIDEO window): P=start/play/pause, R=reset to REST, Q or ESC=quit.',flush=True)
    print(f'MuJoCo markers: BLUE=target trajectory (gaps left open); YELLOW=target; CYAN=actual IK SITE {site_name}; MAGENTA=position error.',flush=True)
    print('Starting in symmetric REST pose; P transitions to the first target, then plays the video.',flush=True)
    # Only the right arm and Dex5 fingers move; the left arm and whole body
    # retain the original rest qpos throughout the transition.
    moving_addrs = np.array(sorted(set(arm_addrs.tolist()+[v[0] for v in bindings.values()])), dtype=int)
    preposition_from = rest_qpos.copy()
    preposition_to = qs[0].copy()

    def display(qpos, video_idx, label):
        data.qpos[:] = qpos
        data.time = times[video_idx]
        mujoco.mj_forward(model,data)
        actual_pos = data.site_xpos[sid].copy()
        actual_rot = data.site_xmat[sid].reshape(3,3).copy()
        add_pose_markers(mujoco, viewer, targets[video_idx], target_rotations[video_idx], actual_pos, actual_rot,
                         path_segments)
        viewer.sync()
        show_video(video_idx, paused=(label != 'PLAYING'))

    # Initial pose must stay at rest, rather than jumping straight to qs[0].
    display(rest_qpos, 0, 'REST')
    last_shown = 0
    try:
        while viewer.is_running():
            key=cv2.waitKey(15) & 0xFF
            if key in (ord('q'),27):
                break
            elif key==ord('p'):
                if phase=='rest':
                    phase='preposition'
                    transition_start=time.monotonic()
                    playing=True
                elif phase=='preposition':
                    # Pause transition by retaining current displayed qpos;
                    # restart transition from that pose on next P.
                    preposition_from=data.qpos.copy()
                    phase='transition_paused'
                    playing=False
                elif phase=='transition_paused':
                    phase='preposition'
                    transition_start=time.monotonic()
                    playing=True
                else:
                    playing=not playing
                    next_advance=time.monotonic()
                    last_shown=-1
            elif key==ord('r'):
                i=0
                phase='rest'
                playing=False
                transition_start=None
                preposition_from=rest_qpos.copy()
                display(rest_qpos, 0, 'REST')
                last_shown=0
                next_advance=time.monotonic()
            if phase=='preposition' and playing:
                fraction = 1.0 if a.preposition_seconds==0 else min(1., (time.monotonic()-transition_start)/a.preposition_seconds)
                smooth = fraction*fraction*(3.-2.*fraction)
                q = rest_qpos.copy()
                q[moving_addrs] = preposition_from[moving_addrs] + smooth*(preposition_to[moving_addrs]-preposition_from[moving_addrs])
                display(q, 0, 'PREPOSITION')
                if fraction>=1.:
                    phase='replay'
                    i=0
                    playing=True
                    next_advance=time.monotonic()
                    last_shown=-1
            elif phase=='replay':
                if i != last_shown:
                    display(qs[i], i, 'PLAYING' if playing else 'PAUSED')
                    last_shown=i
                    delta_t=float(times[i+1]-times[i]) if i+1<len(qs) else 0.
                    if not np.isfinite(delta_t) or delta_t<0:
                        delta_t=1./a.fps
                    # Keep capture intervals cumulative instead of adding the
                    # cost of rendering every frame to the recorded duration.
                    next_advance+=max(.001,delta_t/a.speed)
                if playing and time.monotonic()>=next_advance:
                    if i+1>=len(qs):
                        playing=False
                        last_shown=-1
                        print('End of replay; press R to return to rest or Q to quit.',flush=True)
                    else:
                        i+=1
    finally:
        cap.release()
        cv2.destroyWindow(window)
        viewer.close()
        # close() signals the passive native viewer thread asynchronously.
        # Let it release graphics before Python/Qt interpreter teardown.
        time.sleep(1.)
    print('Replay ended. Inspect sim_quality.csv for position/orientation residuals.',flush=True)


if __name__=='__main__':
    main()
