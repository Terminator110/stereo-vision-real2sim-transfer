#!/usr/bin/env python3
"""DEX-5 video -> G1 right-hand kinematic MuJoCo preview.

Inputs:
 * FoundationStereo/wristband_trial_new/palm_pose_6dof.csv
   SAM3 wristband 3-D surface points and human anatomical palm quaternions.
 * code/../outputs/.../joint_targets.csv from existing 10_g1_dex5_stationary.py
   verified original Dex5 per-finger joint names and thumb fit.
 * models/scenario_g1_dex5_stationary.xml

Transforms table->MuJoCo by a fixed yaw and a diagnostic position translation
anchored to the initial robot hand site. THIS IS NOT a calibrated shared object
workspace. Anatomical hand orientation does not use a first-frame reference.
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
from scipy.spatial.transform import Rotation

ARM = [
    'g1_01_right_shoulder_pitch_joint',
    'g1_01_right_shoulder_roll_joint',
    'g1_01_right_shoulder_yaw_joint',
    'g1_01_right_elbow_joint',
    'g1_01_right_wrist_roll_joint',
    'g1_01_right_wrist_pitch_joint',
    'g1_01_right_wrist_yaw_joint',
]
SITE = 'g1_01_dex5_right_frame'
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


def add_pose_markers(mujoco, viewer, target_position, target_rotation,
                     actual_position, actual_rotation):
    """Show target hand-site pose and actual IK hand-site pose in MuJoCo world.

    YELLOW sphere = target site (currently mapped from wristband surface).
    CYAN sphere = actual g1_01_dex5_right_frame site used by IK.
    MAGENTA line = XYZ residual from target to actual.
    Bright RGB arrows = target site axes; dim RGB arrows = actual site axes.
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
        sphere(target_position, (1.0, .75, .08, 1.0), .017)
        sphere(actual_position, (.05, .9, 1.0, 1.0), .014)
        segment(target_position, actual_position, (1.0, 0.0, .85, 1.0), .004,
                mujoco.mjtGeom.mjGEOM_LINE)
        for origin, rotation, dim in ((target_position, target_rotation, 1.0),
                                      (actual_position, actual_rotation, .45)):
            for axis, color in enumerate(((1.,0.,0.), (0.,1.,0.), (0.,0.,1.))):
                endpoint = origin + rotation[:, axis] * (.075 if dim == 1.0 else .055)
                segment(origin, endpoint, (*[v * dim for v in color], 1.), .004,
                        mujoco.mjtGeom.mjGEOM_ARROW)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path.home()/'physikal_workflow/dex-5'
    ap.add_argument('--palm-csv', type=Path, default=root/'FoundationStereo/wristband_trial_new/palm_pose_6dof.csv')
    ap.add_argument('--finger-csv', type=Path, required=True, help='joint_targets.csv generated by 10_g1_dex5_stationary.py --no-wrist')
    ap.add_argument('--model', type=Path, default=root/'models/scenario_g1_dex5_stationary.xml')
    ap.add_argument('--output', type=Path, default=root/'outputs/dex5_video_to_sim_v1')
    ap.add_argument('--ik-mode', choices=('xyz','pose'), default='xyz',
                    help='xyz: position-first diagnostic; pose: try position + absolute palm orientation')
    ap.add_argument('--table-yaw-deg', type=float, default=-90.,
                    help='Verified table frame -> MuJoCo world yaw for this setup')
    ap.add_argument('--table-origin-world', type=float, nargs=3, metavar=('X','Y','Z'),
                    help='Absolute MuJoCo world position of calibrated table origin (omit for diagnostic initial XYZ anchoring)')
    ap.add_argument('--table-origin-xy-world', type=float, nargs=2,
                    metavar=('X', 'Y'),
                    help='MuJoCo XY of physical table calibration origin; keep original Z anchoring')
    ap.add_argument('--pose-weight', type=float, default=.5, help='Orientation residual weight (pose IK only)')
    ap.add_argument('--position-weight', type=float, default=10., help='Position residual weight')
    ap.add_argument('--max-arm-speed', type=float, default=4., help='Per-joint speed cap, rad/s')
    ap.add_argument('--fps', type=float, default=15., help='Playback FPS fallback if capture timestamps missing')
    ap.add_argument('--speed', type=float, default=.5, help='Visualization playback speed factor')
    ap.add_argument('--start', type=int, default=0, help='Prepared trial sequence index, inclusive')
    ap.add_argument('--end', type=int, help='Prepared trial sequence index, inclusive')
    ap.add_argument('--headless', action='store_true', help='Save trajectory and trace without opening viewer')
    ap.add_argument('--video', type=Path, help='Source camera1.avi to show beside MuJoCo (auto-detected from trial meta.json if omitted)')
    ap.add_argument('--video-width', type=int, default=720, help='Video window width in pixels')
    ap.add_argument('--autoplay', action='store_true', help='Start playing immediately; default starts paused')
    a=ap.parse_args()
    if a.max_arm_speed<=0 or a.fps<=0 or a.position_weight<=0 or not 0<a.speed<=4:
        ap.error('Invalid speed, FPS or position weight')
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
    sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,SITE)
    if sid<0: ap.error('Dex5 site missing: '+SITE)
    arm_ids,arm_addrs=zip(*(get_joint(mujoco,model,n) for n in ARM))
    arm_addrs=np.asarray(arm_addrs,int)
    limits=np.asarray(model.jnt_range[list(arm_ids)],float)
    if np.any(limits[:,1] <= limits[:,0]):
        raise ValueError('Joint limits missing or invalid')
    qcur=data.qpos[arm_addrs].copy()
    initial_hand_pos=data.site_xpos[sid].copy()
    initial_site_rot=data.site_xmat[sid].reshape(3,3).copy()
    janchors=[get_joint(mujoco,model,'g1_01_dex5_right_'+n)[0] for n in FINGER_ANCHORS]
    dex_anat=palm_basis(initial_hand_pos,*[data.xanchor[j].copy() for j in janchors])
    site_to_anat=initial_site_rot.T@dex_anat
    if not np.allclose(site_to_anat.T@site_to_anat,np.eye(3),atol=1e-4):
        raise RuntimeError('Bad Dex5 site-to-anatomical transform')
    yaw=Rotation.from_euler('z',a.table_yaw_deg,degrees=True).as_matrix()
    usable=[]
    for row in palm_rows:
        try: seq=int(row['sequence_frame']);frame=int(row['source_video_frame'])
        except (KeyError, ValueError): continue
        if seq<a.start or (a.end is not None and seq>a.end):continue
        xyz=xyz_from(row,('X_table_m','Y_table_m','Z_table_m'))
        if xyz is None:continue
        q=get_orientation(row)
        try: t=float(row.get('timestamp_s','nan'))
        except ValueError: t=float('nan')
        usable.append((seq,frame,t,xyz,q,row.get('palm_status','?')))
    if not usable:raise RuntimeError('No valid tabletop XYZ in selected palm CSV rows')
    if a.table_origin_world is not None and a.table_origin_xy_world is not None:
        ap.error('Specify only one table-origin option')

    anchor_offset = initial_hand_pos - yaw @ usable[0][3]

    if a.table_origin_xy_world is not None:
        offset = anchor_offset.copy()
        offset[:2] = np.asarray(a.table_origin_xy_world, dtype=float)
        position_mapping = 'TABLE-REGISTERED XY / ORIGINAL ANCHORED Z'
    elif a.table_origin_world is not None:
        offset = np.asarray(a.table_origin_world, dtype=float)
        position_mapping = 'TABLE-REGISTERED XYZ'
    else:
        offset = anchor_offset.copy()
        position_mapping = 'DIAGNOSTIC FIRST-FRAME ANCHOR'
    out=a.output.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    bindings={}
    for key in finger_rows[0].keys():
        if key.startswith(('Pitch_','Roll_')):
            name='g1_01_dex5_right_'+key
        elif key.startswith('g1_01_dex5_right_'):
            name=key
        else:
            continue
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if jid>=0:
            bindings[key]=(int(model.jnt_qposadr[jid]), np.asarray(model.jnt_range[jid],float))
    if len(bindings)<8:
        raise ValueError(f'Only {len(bindings)} Dex5 joints matched finger CSV! Recheck file or model. CSV columns: {list(finger_rows[0])}')
    print(f'Palm frames: {len(usable)}; Dex5 finger bindings: {len(bindings)}',flush=True)
    print(f'Position map: {position_mapping}; offset={offset.round(4)}',flush=True)
    print(f'Orientation: absolute anatomical palm -> Dex5 site; Rz(table yaw)={a.table_yaw_deg} degrees; no first-frame rotation calibration',flush=True)
    print('This is KINEMATIC ONLY: NO robot physics, collision avoidance, object grasp, or contact validation.',flush=True)
    print('Missing/rejected palm orientations HOLD the last valid rotation. Wristband is visible SURFACE, not wrist joint centre.',flush=True)
    print(f'IK mode: {a.ik_mode}',flush=True)
    trace=[];qs=[];times=[];targets=[];target_rotations=[];orientation_valid=0
    last_orientation=None
    last_finger=None
    initial_time=None
    previous_t=None
    for seq,frame,t,xyz_table,r_palm,status in usable:
        if not np.isfinite(t):t=seq/a.fps
        if initial_time is None: initial_time=t
        t=float(t-initial_time)
        if previous_t is None: dt=1./a.fps
        else:dt=max(1./a.fps*.5, min(.2,t-previous_t))
        previous_t=t
        target_pos=yaw@xyz_table + offset
        if r_palm is not None:
            orientation_valid+=1
            last_orientation=yaw@r_palm @ site_to_anat.T
            ori_source='MEASURED'
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
            if a.ik_mode=='pose':
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
        if pos_err<=25 and (a.ik_mode=='xyz' or ori_err<=25):ik_status='TRACKING'
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
                          pose_source=status,orientation_source=ori_source,ik_status=ik_status,
                          position_error_mm=round(pos_err,3),orientation_error_deg=round(ori_err,3),
                          wrist_target_x=target_pos[0],wrist_target_y=target_pos[1],wrist_target_z=target_pos[2],
                          wrist_actual_x=p[0],wrist_actual_y=p[1],wrist_actual_z=p[2],
                          finger_joints_applied=applied,thumb_state=last_finger.get('thumb_state','?') if last_finger else 'NO_FINGER'))
        if seq%10==0 or seq==usable[-1][0]:
            print(f'{seq+1:3d}: ARM={ik_status:11} pos={pos_err:5.1f}mm rot={ori_err:5.1f}deg palm={ori_source:12} finger={applied}',flush=True)
    np.savez_compressed(out/'sim_qpos.npz',qpos=np.asarray(qs),time_s=np.asarray(times),
                        source_frames=np.asarray([v[1] for v in usable]),target_pos=np.asarray(targets),
                        target_quat_xyzw=Rotation.from_matrix(np.asarray(target_rotations)).as_quat())
    with (out/'sim_quality.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(trace[0].keys()));w.writeheader();w.writerows(trace)
    summary=dict(frames=len(qs),measured_palm_orientations=orientation_valid,
                 with_finger_targets=sum(v['finger_joints_applied']>=8 for v in trace),
                 position_mapping=position_mapping,table_origin_to_world_translation=offset.tolist(),
                 ik_mode=a.ik_mode,statuses={s:sum(v['ik_status']==s for v in trace) for s in ('TRACKING','APPROX','LARGE_ERROR')},
                 median_position_error_mm=float(np.median([v['position_error_mm'] for v in trace])),
                 median_orientation_error_deg=float(np.median([v['orientation_error_deg'] for v in trace])),
                 physics=False,collision_checked=False,grasp_tested=False)
    (out/'sim_report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Simulation trace:',out/'sim_quality.csv',flush=True)
    print('Qpos trajectory:',out/'sim_qpos.npz',flush=True)
    print('Report:',out/'sim_report.json',flush=True)
    print('Statuses:',summary['statuses'],flush=True)
    if a.headless:return
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
        wanted = int(usable[idx][1])
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
            f"{'PAUSED' if paused else 'PLAYING'}  [{idx+1}/{len(qs)}]  source frame {usable[idx][1]}",
            f"IK {status['ik_status']}  pos {status['position_error_mm']:.1f}mm  rot {status['orientation_error_deg']:.1f}deg",
            f"Palm {status['orientation_source']}   P: play/pause   R: restart   Q: quit",
            "MuJoCo: YELLOW=target wristband  CYAN=actual Dex5 IK site  PINK=line error",
        ]
        for j, line in enumerate(lines):
            y = 28 + j*27
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, vis)

    data.qpos[:]=qs[0]
    mujoco.mj_forward(model,data)
    viewer=mjv.launch_passive(model,data)
    viewer.cam.lookat[:]=data.site_xpos[sid]
    viewer.cam.distance=1.5
    viewer.cam.azimuth=140
    viewer.cam.elevation=-18
    playing = bool(a.autoplay)
    i = 0
    last_shown = -1
    next_advance = time.monotonic()
    print(f'Human video: {video_path}',flush=True)
    print('Playback controls (focus VIDEO window): P=play/pause, R=restart, Q or ESC=quit.',flush=True)
    print('MuJoCo markers: YELLOW=target; CYAN=actual IK SITE g1_01_dex5_right_frame; MAGENTA=position error.',flush=True)
    print('Paused at frame 0 until P is pressed.' if not playing else 'Autoplay enabled.',flush=True)
    try:
        while viewer.is_running():
            if i != last_shown:
                data.qpos[:]=qs[i]
                data.time=times[i]
                mujoco.mj_forward(model,data)
                actual_pos = data.site_xpos[sid].copy()
                actual_rot = data.site_xmat[sid].reshape(3, 3).copy()
                add_pose_markers(mujoco, viewer, targets[i], target_rotations[i],
                                 actual_pos, actual_rot)
                viewer.sync()
                show_video(i, not playing)
                last_shown = i
            key=cv2.waitKey(15) & 0xFF
            if key in (ord('q'),27):
                break
            elif key==ord('p'):
                playing=not playing
                next_advance=time.monotonic()
                last_shown=-1
            elif key==ord('r'):
                i=0
                playing=False
                last_shown=-1
                next_advance=time.monotonic()
            if playing and time.monotonic()>=next_advance:
                if i+1>=len(qs):
                    playing=False
                    last_shown=-1
                    print('End of replay; press R to restart or Q to quit.',flush=True)
                else:
                    delta_t=float(times[i+1]-times[i])
                    if not np.isfinite(delta_t) or delta_t<=0:
                        delta_t=1./a.fps
                    next_advance=time.monotonic()+max(.005,min(.2,delta_t/a.speed))
                    i+=1
    finally:
        cap.release()
        cv2.destroyWindow(window)
        viewer.close()
    print('Replay ended. Inspect sim_quality.csv for position/orientation residuals.',flush=True)


if __name__=='__main__':
    main()
