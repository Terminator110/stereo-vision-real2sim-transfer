#!/usr/bin/env python3
"""Live stereo MediaPipe wrist + fingers -> G1/Dex5 MuJoCo wrist-site IK.

IMPORTANT: MediaPipe landmark 0 is an *anatomical wrist estimate*, NOT the
SAM3 wristband-surface centroid used by the offline 150-frame pipeline.
This diagnostic deliberately requires both calibrated cameras; no fake depth,
no physical robot commands, no collisions/contact/physics stepping.

Keys in Camera1 window: P start/pause, R rest/reset, Q/Esc exit.
"""
from __future__ import annotations
import argparse, csv, importlib.util, time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import mujoco
import mediapipe as mp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f'Unable to import {path}')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

old = load_module('_dex5_live_stereo_12', HERE/'12_stereo_g1_dex5.py')
ref = load_module('_dex5_live_wrist_ik_tools', HERE/'dex5_video_to_sim_wrist_site.py')

def open_cam(dev, size):
    c=cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not c.isOpened(): raise RuntimeError(f'Cannot open camera /dev/video{dev}')
    c.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    c.set(cv2.CAP_PROP_FRAME_WIDTH,int(size[0]))
    c.set(cv2.CAP_PROP_FRAME_HEIGHT,int(size[1]))
    c.set(cv2.CAP_PROP_BUFFERSIZE,1)
    return c

def make_hand(task):
    opts=mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,min_hand_detection_confidence=.45,
        min_hand_presence_confidence=.45,min_tracking_confidence=.45)
    return mp.tasks.vision.HandLandmarker.create_from_options(opts)

def run():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cam1',type=int,default=2)
    ap.add_argument('--cam2',type=int,default=0)
    ap.add_argument('--no-rotate-cam2',action='store_true',help='Disable required 180 degree live rotation if your camera was physically corrected')
    ap.add_argument('--calib',type=Path,default=HERE/'stereo/calibration/parallel_results_v3/stereo_calibration.npz')
    ap.add_argument('--table-frame',type=Path,default=HERE/'stereo/calibration/parallel_results_v3/table_frame_multi.npz')
    ap.add_argument('--hand-model',type=Path,default=ROOT/'models/mediapipe/hand_landmarker.task')
    ap.add_argument('--model',type=Path,default=ROOT/'models/scenario_g1_dex5_stationary.xml')
    ap.add_argument('--table-origin-world',nargs=3,type=float,default=(.385,-.120,.832))
    ap.add_argument('--table-yaw-deg',type=float,default=-90.)
    ap.add_argument('--max-reproj',type=float,default=5.)
    ap.add_argument('--max-grab-delta-ms',type=float,default=80.)
    ap.add_argument('--ik-mode',choices=('xyz','pose'),default='xyz',help='Start with XYZ to isolate frame mapping from palm orientation')
    ap.add_argument('--pose-weight',type=float,default=.5)
    ap.add_argument('--position-weight',type=float,default=10.)
    ap.add_argument('--max-arm-speed',type=float,default=2.)
    ap.add_argument('--robot-alpha',type=float,default=.32)
    ap.add_argument('--max-frames',type=int,default=0)
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/live_stereo_wrist_ik.csv')
    a=ap.parse_args()
    if a.cam1==a.cam2: ap.error('Two distinct cameras are required')
    if not (.05<=a.robot_alpha<=1): ap.error('Robot alpha must be between .05 and 1')
    if a.max_arm_speed<=0 or a.max_reproj<=0: ap.error('Limits must be positive')
    if not np.isfinite(a.table_origin_world).all():ap.error('Invalid table world origin')
    cal,size,Rct,tct=old.load_calibration(a.calib,a.table_frame)
    m=mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    d=mujoco.MjData(m)
    d.qpos[:]=m.qpos0
    for name,val in [('g1_01_right_shoulder_roll_joint',-.25),('g1_01_right_elbow_joint',.6)]:
        jid,adr=ref.get_joint(mujoco,m,name)
        d.qpos[adr]=np.clip(val,*m.jnt_range[jid])
    mujoco.mj_forward(m,d)
    rest=d.qpos.copy()
    sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'g1_01_right_wrist_ik_site')
    palm_sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'g1_01_dex5_right_frame')
    if min(sid,palm_sid)<0:
        raise RuntimeError('Wrist site or Dex5 site missing: ensure scene loads your modified wrist-site XML')
    jids,addrs=zip(*(ref.get_joint(mujoco,m,n) for n in ref.ARM))
    addrs=np.asarray(addrs,int)
    limits=np.asarray(m.jnt_range[list(jids)],float)
    qcur=d.qpos[addrs].copy()
    anchors=[ref.get_joint(mujoco,m,'g1_01_dex5_right_'+n)[0] for n in ref.FINGER_ANCHORS]
    dex_anat=ref.palm_basis(d.site_xpos[palm_sid].copy(),*[d.xanchor[j].copy() for j in anchors])
    site_to_anat=d.site_xmat[sid].reshape(3,3).T@dex_anat
    yaw=Rotation.from_euler('z',a.table_yaw_deg,degrees=True).as_matrix()
    origin=np.asarray(a.table_origin_world,float)
    retarget=old.StereoRetargeter(m)
    # Reuse existing original Dex5 finger retargeter; no synthetic finger angles.
    finger_addrs=np.array(sorted({binding[0] for binding in retarget.bindings.values()}),dtype=int)
    for i in range(m.ngeom):
        b=int(m.geom_bodyid[i]); robot=False
        while b>0:
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) or ''
            if name.startswith('g1_01') or 'dex5' in name.lower():robot=True;break
            b=int(m.body_parentid[b])
        if robot:m.geom_rgba[i,3]=min(m.geom_rgba[i,3],a.robot_alpha)
    c1=open_cam(a.cam1,size)
    c2=open_cam(a.cam2,size)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    from mujoco import viewer as mjviewer
    window='LIVE CAM1 | P:track/pause R:rest Q:quit'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    status='REST'; playing=False; last_orientation=None; last_ms=-1
    t0=time.monotonic();previous=time.monotonic();valid=0;count=0
    # Don't move the arm on invalid frames; do not invent depth.
    try:
      with make_hand(a.hand_model) as h1,make_hand(a.hand_model) as h2, mjviewer.launch_passive(m,d) as viewer, a.output.open('w',newline='') as handle:
        cols=['frame','elapsed_s','status','active','stereo_status','grab_skew_ms',
              'wrist_table_x','wrist_table_y','wrist_table_z','target_x','target_y','target_z',
              'actual_x','actual_y','actual_z','pos_error_mm','rot_error_deg','finger_groups']
        writer=csv.DictWriter(handle,fieldnames=cols);writer.writeheader()
        viewer.cam.lookat[:]=[.38,-.14,.84];viewer.cam.distance=1.45;viewer.cam.azimuth=140;viewer.cam.elevation=-18
        print('LIVE CALIBRATED CAMERAS: Camera1=/dev/video%d, Camera2=/dev/video%d' %(a.cam1,a.cam2),flush=True)
        print('MediaPipe landmark 0 anatomical wrist; NOT SAM3 wristband surface centroid.',flush=True)
        print('Table origin XYZ:',origin,'; yaw:',a.table_yaw_deg,'; IK: right wrist-yaw joint site',flush=True)
        print('Starts REST: click CAM1 window, P toggles active tracking, R resets to REST, Q quits.',flush=True)
        print('No real robot control. Kinematic MuJoCo only. No collision avoidance.',flush=True)
        while viewer.is_running() and (a.max_frames==0 or count<a.max_frames):
            now=time.monotonic(); dt=np.clip(now-previous,.001,.2);previous=now
            ok1=c1.grab();t1=time.monotonic();ok2=c2.grab();t2=time.monotonic()
            if not(ok1 and ok2):print('Camera grab failed');break
            ok1,f1=c1.retrieve();ok2,f2=c2.retrieve()
            if not(ok1 and ok2):print('Camera retrieve failed');break
            if not a.no_rotate_cam2:f2=cv2.rotate(f2,cv2.ROTATE_180)
            if (f1.shape[1],f1.shape[0])!=size or (f2.shape[1],f2.shape[0])!=size:
                raise RuntimeError('Unexpected camera resolution. Stereo calibration is '+str(size))
            ms=max(last_ms+1,int(1000*(now-t0)));last_ms=ms
            img1=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(f1,cv2.COLOR_BGR2RGB)))
            img2=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(f2,cv2.COLOR_BGR2RGB)))
            pts1,_=old.image_landmarks(h1.detect_for_video(img1,ms),size)
            pts2,_=old.image_landmarks(h2.detect_for_video(img2,ms),size)
            skew=1000*(t2-t1)
            result=old.reject('CAMERA_GRAB_SKEW') if skew>a.max_grab_delta_ms else old.reconstruct_hand(pts1,pts2,cal,Rct,tct,a.max_reproj)
            p3=result['points_table']; groups=result['groups']
            wrist=None;target=None;actual=None;err=float('nan');rerr=float('nan')
            good_wrist=bool(p3 is not None and groups.get('wrist') and np.isfinite(p3[0]).all())
            if good_wrist:
                wrist=p3[0].copy()
                target=yaw@wrist+origin
                valid+=1
            # Only use measured 3D points. Hold last frame on missing observations.
            if playing and good_wrist:
                if a.ik_mode=='pose':
                    try:
                        hand_R=ref.palm_basis(*[p3[j] for j in (0,5,9,17)])
                        last_orientation=yaw@hand_R@site_to_anat.T
                    except (FloatingPointError,ValueError):pass
                old_q=qcur.copy()
                def fk(q):
                    d.qpos[addrs]=q
                    mujoco.mj_forward(m,d)
                    return d.site_xpos[sid].copy(),d.site_xmat[sid].reshape(3,3).copy()
                def residual(q):
                    p,R=fk(q)
                    chunks=[a.position_weight*(p-target)]
                    if a.ik_mode=='pose' and last_orientation is not None:
                        chunks.append(a.pose_weight*Rotation.from_matrix(last_orientation@R.T).as_rotvec())
                    chunks.append(.025*(q-old_q))
                    return np.concatenate(chunks)
                fit=least_squares(residual,old_q,bounds=(limits[:,0]+1e-7,limits[:,1]-1e-7),max_nfev=35)
                change=np.clip(fit.x-old_q,-a.max_arm_speed*dt,a.max_arm_speed*dt)
                qcur=np.clip(old_q+change,limits[:,0],limits[:,1])
                d.qpos[addrs]=qcur
                # Existing 12_stereo retargeter computes finger targets from 3D landmark chains.
                # Its independent wrist angles are intentionally NOT applied.
                if p3 is not None:
                    try:
                        finger_pose,fstate=retarget.update(p3,now)
                        d.qpos[finger_addrs]=finger_pose[finger_addrs]
                    except (FloatingPointError,ValueError):
                        pass
                mujoco.mj_forward(m,d)
                actual=d.site_xpos[sid].copy()
                err=1000*np.linalg.norm(actual-target)
                if last_orientation is not None:
                    actual_R=d.site_xmat[sid].reshape(3,3)
                    rerr=np.degrees(Rotation.from_matrix(last_orientation@actual_R.T).magnitude())
                status='TRACKING'
            elif playing:
                status='HOLD: INVALID STEREO'
            else:
                status='PAUSED'
            # Draw the exact current target and actual wrist-joint site.
            if target is not None:
                p_act=d.site_xpos[sid].copy()
                R_act=d.site_xmat[sid].reshape(3,3).copy()
                R_tar=last_orientation if last_orientation is not None else R_act
                ref.add_pose_markers(mujoco,viewer,target,R_tar,p_act,R_act)
            else:
                with viewer.lock(): viewer.user_scn.ngeom=0
            viewer.sync()
            vis=f1.copy()
            if pts1 is not None:
                uv=tuple(np.rint(pts1[0]).astype(int))
                cv2.circle(vis,uv,9,(0,255,255),2)
                cv2.putText(vis,'MediaPipe WRIST (not band centroid)',(uv[0]+12,uv[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),2)
            lines=[f'LIVE G1 wrist IK: {status} | {a.ik_mode} | {result["status"]}',
                   f'Frame {count}  Valid wrist {valid}/{count+1}  skew {skew:.0f}ms',
                   'YELLOW=target | CYAN=actual in MuJoCo | P toggle | R reset | Q quit']
            if target is not None:
                lines.append(f'Target XYZ world: {target[0]:+.3f} {target[1]:+.3f} {target[2]:+.3f}m | error {err:.1f}mm')
            for j,line in enumerate(lines):
                y=28+27*j
                cv2.putText(vis,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(vis,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            cv2.imshow(window,cv2.resize(vis,(960,round(vis.shape[0]*960/vis.shape[1]))))
            key=cv2.waitKey(1)&255
            if key in (ord('q'),27):break
            if key==ord('p'):
                playing=not playing
                print('LIVE TRACKING' if playing else 'PAUSED',flush=True)
            if key==ord('r'):
                playing=False;status='REST';last_orientation=None
                d.qpos[:]=rest;qcur=rest[addrs].copy();mujoco.mj_forward(m,d)
                # Restart finger retargeter filters, too.
                retarget=old.StereoRetargeter(m)
            writer.writerow(dict(frame=count,elapsed_s=round(now-t0,3),status=status,
                active=int(playing),stereo_status=result['status'],grab_skew_ms=round(skew,2),
                wrist_table_x=wrist[0] if wrist is not None else '',
                wrist_table_y=wrist[1] if wrist is not None else '',
                wrist_table_z=wrist[2] if wrist is not None else '',
                target_x=target[0] if target is not None else '',
                target_y=target[1] if target is not None else '',
                target_z=target[2] if target is not None else '',
                actual_x=actual[0] if actual is not None else '',
                actual_y=actual[1] if actual is not None else '',
                actual_z=actual[2] if actual is not None else '',
                pos_error_mm=round(err,2) if np.isfinite(err) else '',
                rot_error_deg=round(rerr,2) if np.isfinite(rerr) else '',
                finger_groups=sum(bool(v) for k,v in groups.items() if k!='wrist')))
            if count%10==0:
                print(f'frame {count}: {status}; stereo={result["status"]}; target={None if target is None else target.round(3)}; pos_err={err:.1f}mm',flush=True)
            count+=1
    finally:
        c1.release();c2.release();cv2.destroyAllWindows()
    print('Saved:',a.output,flush=True)

if __name__=='__main__':run()
