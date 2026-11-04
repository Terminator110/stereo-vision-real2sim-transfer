"""Stationary G1 + Dex5: right-hand webcam fingers and calibrated wrist IK.

Kinematic test only: no mj_step, walking policy, ROS, or hardware commands.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import time
import threading

import cv2
import mediapipe as mp
import mujoco
import numpy as np
from thumb_retarget import ThumbRetargeter
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dex5_live',Path(__file__).with_name('08_live_webcam_dex5.py'))
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)
retarget = live.retarget
WRIST_NAMES = [f'g1_01_right_wrist_{axis}_joint' for axis in ('roll','pitch','yaw')]


def address(model,name):
    joint = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    if joint < 0:
        raise ValueError(f'Missing joint {name}')
    return int(model.jnt_qposadr[joint])


def setup_pose(model):
    # Fixed upper arms and modest elbow bend; these do not follow the camera.
    for side,roll in [('left',0.25),('right',-0.25)]:
        model.qpos0[address(model,f'g1_01_{side}_shoulder_roll_joint')] = roll
        model.qpos0[address(model,f'g1_01_{side}_elbow_joint')] = 0.6
    return model.qpos0.copy()


def right_bindings(model):
    result = {}
    for group in retarget.FINGERS.values():
        for name in group:
            actual = 'g1_01_dex5_right_'+name
            joint = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,actual)
            if joint < 0:
                raise ValueError(f'Missing right Dex5 joint {actual}')
            result[name] = (int(model.jnt_qposadr[joint]),*map(float,model.jnt_range[joint]))
    return result


def palm_frame(wrist,index,middle,pinky):
    forward = retarget.unit(middle-wrist)
    across = index-pinky
    across = retarget.unit(across-forward*np.dot(across,forward))
    normal = retarget.unit(np.cross(across,forward))
    return np.column_stack([across,forward,normal])


class WristTracker:
    def __init__(self,model,cutoff=2.5,max_velocity=1.5):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = model.qpos0
        self.ids = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in WRIST_NAMES]
        if min(self.ids) < 0:
            raise ValueError('Expected three existing G1 wrist joints')
        self.addresses = np.array([model.jnt_qposadr[j] for j in self.ids])
        self.bounds = model.jnt_range[self.ids].copy()
        self.site = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'g1_01_dex5_right_frame')
        if self.site < 0:
            raise ValueError('Missing Dex5 palm frame site')
        mujoco.mj_forward(model,self.data)
        neutral = self.data.site_xmat[self.site].reshape(3,3).copy()
        joints = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'g1_01_dex5_right_'+n)
                  for n in ('Roll_21R','Roll_31R','Roll_51R')]
        anatomical = palm_frame(self.data.site_xpos[self.site],*[self.data.xanchor[j] for j in joints])
        self.robot_basis = neutral.T @ anatomical
        self.previous = self.data.qpos[self.addresses].copy()
        self.cutoff,self.max_velocity = cutoff,max_velocity
        self.last_time = None
        self.reset()

    def orientation(self,q):
        self.data.qpos[self.addresses] = q
        mujoco.mj_kinematics(self.model,self.data)
        return self.data.site_xmat[self.site].reshape(3,3).copy()

    def reset(self):
        self.reference = None
        self.pending = []
        self.robot_reference = self.orientation(self.previous)

    def update(self,points,now):
        dt = 1/30 if self.last_time is None else float(np.clip(now-self.last_time,0,0.1))
        self.last_time = now
        if points is None:
            self.pending.clear()
            return self.previous.copy(),'LOST_HOLD',None
        try:
            basis = palm_frame(points[0],points[5],points[9],points[17])
        except ValueError:
            self.pending.clear()
            return self.previous.copy(),'DEGENERATE_HOLD',None
        if self.reference is None:
            self.pending.append(basis)
            if len(self.pending) < 5:
                return self.previous.copy(),'CALIBRATING',None
            self.reference = Rotation.from_matrix(np.stack(self.pending)).mean().as_matrix()
        relative = self.reference.T @ basis
        target = self.robot_reference @ self.robot_basis @ relative @ self.robot_basis.T
        previous = self.previous.copy()
        def residual(q):
            error = Rotation.from_matrix(target.T @ self.orientation(q)).as_rotvec()
            return np.r_[error,0.02*(q-previous)]
        result = least_squares(residual,previous,bounds=(self.bounds[:,0],self.bounds[:,1]),
                               max_nfev=15,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        if not np.isfinite(result.x).all():
            return previous,'SOLVER_HOLD',None
        alpha = -np.expm1(-2*np.pi*self.cutoff*dt)
        change = np.clip(alpha*(result.x-previous),-self.max_velocity*dt,self.max_velocity*dt)
        self.previous = np.clip(previous+change,self.bounds[:,0],self.bounds[:,1])
        error = float(np.linalg.norm(Rotation.from_matrix(target.T @ self.orientation(self.previous)).as_rotvec())*180/np.pi)
        return self.previous.copy(),'LIMIT_OR_LAG' if error > 20 else 'TRACKING',error


def demo_pose(model,neutral,t):
    pose = neutral.copy()
    curl = 0.5-0.5*np.cos(2*np.pi*t/3)
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)
        if name.startswith('g1_01_dex5_') and ('_Pitch_' in name or '_Roll_12' in name):
            low,high = model.jnt_range[j]
            closed = low if abs(low)>abs(high) else high
            pose[model.jnt_qposadr[j]] = closed*curl
    for side in ('left','right'):
        for axis,amplitude in [('roll',0.35),('pitch',0.25),('yaw',0.3)]:
            pose[address(model,f'g1_01_{side}_wrist_{axis}_joint')] = amplitude*np.sin(2*np.pi*t/4)
    return pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--demo',action='store_true',help='Both-hand and wrist sweep, no camera')
    sources.add_argument('--video',type=Path,help='Recorded-video regression instead of webcam')
    sources.add_argument('--camera',type=int,default=0)
    parser.add_argument('--model',type=Path,default=ROOT/'models/scenario_g1_dex5_stationary.xml')
    parser.add_argument('--hand-model',type=Path,default=ROOT/'models/mediapipe/hand_landmarker.task')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/datetime.now().strftime('g1_dex5_%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--view',choices=['full','right-hand'],default='full')
    parser.add_argument('--no-wrist',action='store_true')
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    if args.headless and not args.video and not args.max_frames:
        parser.error('Headless demo/webcam requires --max-frames')
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    neutral = setup_pose(model)
    bindings = right_bindings(model)
    fingers = live.LiveFilter(model,bindings)
    thumb = ThumbRetargeter(model,bindings)
    wrist = WristTracker(model)
    data = mujoco.MjData(model)
    retarget.apply_pose(model,data,neutral)
    args.output.mkdir(parents=True,exist_ok=False)
    viewer,cap,tracker = None,None,None
    viewer_threads = []
    counts,poses,errors = Counter(),[],[]
    thumb_counts = Counter()
    window = 'G1 + Dex5 | RIGHT webcam'
    index,previous_ms = 0,-1
    start = time.monotonic()
    try:
        if not args.demo:
            if not args.hand_model.is_file():
                raise FileNotFoundError(args.hand_model)
            cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
            if not cap.isOpened():
                raise RuntimeError('Cannot open camera/video; check --camera index and permissions')
            if not args.video:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            fps = cap.get(cv2.CAP_PROP_FPS) if args.video else 30
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('Invalid video FPS')
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(args.hand_model)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=1,
                min_hand_detection_confidence=0.5,min_hand_presence_confidence=0.5,min_tracking_confidence=0.5)
            tracker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        if not args.headless:
            from mujoco import viewer as mj_viewer
            existing_threads = set(threading.enumerate())
            viewer = mj_viewer.launch_passive(model,data)
            viewer_threads = [thread for thread in threading.enumerate() if thread not in existing_threads]
            viewer.cam.lookat[:] = data.site_xpos[wrist.site] if args.view == 'right-hand' else [0,0,0.85]
            viewer.cam.distance = 0.65 if args.view == 'right-hand' else 2.4
            viewer.cam.azimuth,viewer.cam.elevation = 135,-15
            if not args.demo:
                cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        print('Stationary kinematic test: G1 body fixed; NO walking, dynamics stepping, or hardware control.',flush=True)
        if not args.demo:
            print('One RIGHT hand. Hold neutral for 5 detected frames; R in video window recalibrates wrist.',flush=True)
        with (args.output/'joint_targets.csv').open('w',newline='') as log:
            writer = csv.writer(log)
            writer.writerow(['frame','source_time_s','finger_state','wrist_state','orientation_error_deg','raw_label','thumb_state','thumb_ik_fit_mm',*WRIST_NAMES,*bindings])
            while viewer is None or viewer.is_running():
                begin = time.monotonic()
                if args.demo:
                    now = index/30 if args.headless else begin-start
                    pose = demo_pose(model,neutral,now)
                    finger_state,wrist_state,error,label = 'DEMO','DEMO',None,'N/A'
                    thumb_state,thumb_error = 'DEMO',None
                else:
                    ok,frame = cap.read()
                    if not ok:
                        if args.video:
                            break
                        raise RuntimeError('Camera stopped delivering frames')
                    now = index/fps if args.video else time.monotonic()-start
                    ms = max(previous_ms+1,round(now*1000)); previous_ms = ms
                    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    infer_start = time.monotonic()
                    result = tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)),ms)
                    inference_ms = (time.monotonic()-infer_start)*1000
                    angles,landmarks,label = live.read_hand(result)
                    points = np.array([[p.x,p.y,p.z] for p in result.hand_world_landmarks[0]]) if angles is not None else None
                    thumb_angles,thumb_state,thumb_error = thumb.update(points)
                    if angles is not None:
                        angles.update(thumb_angles)
                    pose,finger_state = fingers.update(angles,now)
                    q,wrist_state,error = wrist.update(points,now) if not args.no_wrist else (neutral[wrist.addresses],'DISABLED',None)
                    pose[wrist.addresses] = q
                retarget.apply_pose(model,data,pose)
                data.time = now
                if not np.isfinite(data.xpos).all():
                    raise RuntimeError('Nonfinite forward kinematics')
                poses.append(pose.copy())
                counts[wrist_state] += 1
                thumb_counts[thumb_state] += 1
                if error is not None:
                    errors.append(error)
                writer.writerow([index,now,finger_state,wrist_state,error,label,thumb_state,thumb_error,*pose[wrist.addresses],*[pose[a] for a,_,_ in bindings.values()]])
                if viewer is not None:
                    viewer.sync()
                    if not args.demo:
                        preview = live.preview_frame(frame,landmarks,finger_state,label,index,True,inference_ms,thumb_state)
                        preview = cv2.copyMakeBorder(preview,0,65,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                        if args.no_wrist:
                            calibration_text = 'Calibration disabled'
                        elif wrist.reference is None:
                            calibration_text = f'Calibration started ({len(wrist.pending)}/5)'
                        else:
                            calibration_text = 'Calibration complete'
                        calibration_color = (0,210,0) if wrist.reference is not None and not args.no_wrist else (0,220,255)
                        cv2.putText(preview,calibration_text,(8,preview.shape[0]-42),cv2.FONT_HERSHEY_SIMPLEX,0.55,calibration_color,1)
                        text = f'Wrist: {wrist_state}' + (f' | error {error:.1f} deg' if error is not None else '')
                        cv2.putText(preview,text,(8,preview.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,220,255),1)
                        cv2.imshow(window,preview)
                        key = cv2.waitKey(1) & 0xff
                        if key == ord('r'):
                            wrist.reset()
                        if key in (27,ord('q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE) < 1:
                            index += 1
                            break
                    time.sleep(max(0,1/(fps if args.video else 30)-(time.monotonic()-begin)))
                index += 1
                if args.max_frames is not None and index >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print('Stopped.',flush=True)
    finally:
        if cap is not None:
            cap.release()
        if viewer is not None:
            viewer.close()
            # The passive viewer's daemon thread must finish before GL is unloaded.
            for thread in viewer_threads:
                thread.join(timeout=5)
        if not args.headless:
            cv2.destroyAllWindows()
        if tracker is not None:
            tracker.close()
    np.savez_compressed(args.output/'trajectory.npz',qpos=np.asarray(poses),neutral=neutral)
    controlled = set(wrist.addresses.tolist()) | {a for a,_,_ in bindings.values()}
    if args.demo:
        controlled |= {int(model.jnt_qposadr[j]) for j in range(model.njnt)
                       if 'dex5_' in mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or '_wrist_' in mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)}
    fixed = sorted(set(range(model.nq))-controlled)
    stationary = not poses or bool(np.allclose(np.asarray(poses)[:,fixed],neutral[fixed],atol=1e-12))
    report = dict(frames=len(poses),model=str(args.model.resolve()),mode='demo' if args.demo else ('video' if args.video else 'webcam'),
                  nq=int(model.nq),g1_body_stationary=stationary,wrist_states=dict(counts),
                  thumb_ik_states=dict(thumb_counts),thumb_method='Four-joint palm-relative bounded IK with persistent-jump gate',
                  orientation_error_median_deg=float(np.median(errors)) if errors else None,
                  max_orientation_error_deg=max(errors) if errors else None,walking_tested=False,
                  limitations='Fixed base; direct qpos/mj_forward only. No dynamics/standing balance/grasp/contact/hardware validation. Wrist is relative palm orientation, not forearm-measured anatomical wrist motion.',
                  privacy='Local inference; no camera video/images saved')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'Frames={len(poses)}; G1 body stationary={stationary}; output={args.output}',flush=True)


if __name__ == '__main__':
    main()
