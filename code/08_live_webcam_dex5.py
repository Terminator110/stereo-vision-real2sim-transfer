"""Local webcam -> MediaPipe -> Dex5 RIGHT kinematic visualization only."""
import argparse
from collections import Counter, deque
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import time

import cv2
import mediapipe as mp
import mujoco
import numpy as np
from thumb_retarget import ThumbRetargeter

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dex5_retarget',Path(__file__).with_name('07_retarget_dex5_right.py'))
retarget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retarget)
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
               (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


class LiveFilter:
    """Median + low-pass + slew limiting, including bounded loss/reacquisition."""
    def __init__(self, model, bindings, cutoff=3.0, thumb_cutoff=2.0, max_velocity=3.0,
                 thumb_velocity=2.0, hold_s=0.5, gain=1.0):
        self.bindings = bindings
        self.neutral = model.qpos0.copy()
        self.low = self.neutral.copy()
        self.high = self.neutral.copy()
        self.cutoff = np.full(model.nq,cutoff)
        self.velocity = np.full(model.nq,max_velocity)
        for name,(address,low,high) in bindings.items():
            self.neutral[address] = np.clip(0,low,high)
            self.low[address],self.high[address] = low,high
            if name in retarget.FINGERS['thumb']:
                self.cutoff[address] = thumb_cutoff
                self.velocity[address] = thumb_velocity
        self.pose = self.neutral.copy()
        self.history = deque(maxlen=3)
        self.last_seen = None
        self.last_time = None
        self.hold_s,self.gain = hold_s,gain

    def update(self, angles, now):
        dt = 1/30 if self.last_time is None else np.clip(now-self.last_time,0,0.1)
        self.last_time = now
        if angles is not None:
            target = self.neutral.copy()
            for name,value in angles.items():
                if not np.isfinite(value):
                    raise ValueError('Nonfinite joint target')
                address,low,high = self.bindings[name]
                target[address] = np.clip(self.gain*value,low,high)
            self.history.append(target)
            target = np.median(np.stack(self.history),axis=0)
            self.last_seen = now
            status = 'TRACKING RIGHT (assumed)'
        else:
            self.history.clear()
            if self.last_seen is not None and now-self.last_seen <= self.hold_s:
                return self.pose.copy(),'TRACKING LOST - HOLD'
            target = self.neutral.copy()
            status = 'TRACKING LOST - NEUTRAL'
        alpha = -np.expm1(-2*np.pi*self.cutoff*dt)
        delta = alpha*(target-self.pose)
        self.pose += np.clip(delta,-self.velocity*dt,self.velocity*dt)
        self.pose = np.clip(self.pose,self.low,self.high)
        return self.pose.copy(),status


def read_hand(result):
    if not result.hand_world_landmarks or not result.hand_landmarks:
        return None,None,'Unknown'
    world = np.array([[p.x,p.y,p.z] for p in result.hand_world_landmarks[0]])
    label = 'Unknown'
    if result.handedness and result.handedness[0]:
        label = result.handedness[0][0].category_name
    try:
        angles = retarget.landmark_angles(world)
    except ValueError:
        return None,None,label
    return angles,result.hand_landmarks[0],label


def preview_frame(frame,landmarks,status,label,index,mirror,inference_ms,thumb_status=None):
    preview = frame.copy()
    h,w = preview.shape[:2]
    if landmarks is not None:
        points = [(int(p.x*w),int(p.y*h)) for p in landmarks]
        for a,b in CONNECTIONS:
            cv2.line(preview,points[a],points[b],(0,210,0),2)
        for i,point in enumerate(points):
            cv2.circle(preview,point,4,(0,180,255) if 1 <= i <= 4 else (0,220,0),-1)
    if mirror:
        preview = cv2.flip(preview,1)
    scale = min(960/w,720/h,1)
    preview = cv2.resize(preview,(max(1,round(w*scale)),max(1,round(h*scale))))
    banner = np.zeros((110 if thumb_status else 80,preview.shape[1],3),dtype=np.uint8)
    color = (0,210,0) if status.startswith('TRACKING RIGHT') else (0,170,255)
    cv2.putText(banner,status,(8,25),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
    cv2.putText(banner,f'{index} | raw label: {label} | infer: {inference_ms:.0f} ms',
                (8,55),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
    if thumb_status:
        cv2.putText(banner,f'Thumb: {thumb_status}',(8,88),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,220,255),1)
    return np.vstack([banner,preview])


def positive(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Value must be positive and finite')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--camera',type=int,default=0)
    source.add_argument('--video',type=Path,help='Offline regression test instead of accessing a webcam')
    parser.add_argument('--model',type=Path,default=retarget.DEFAULT_MODEL)
    parser.add_argument('--hand-model',type=Path,default=ROOT/'models/mediapipe/hand_landmarker.task')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/datetime.now().strftime('test8_live_%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--max-seconds',type=positive)
    parser.add_argument('--no-mirror-preview',action='store_true')
    parser.add_argument('--smoothing-hz',type=positive,default=3.0)
    parser.add_argument('--thumb-smoothing-hz',type=positive,default=2.0)
    parser.add_argument('--max-velocity',type=positive,default=3.0,help='Joint-target slew limit in rad/s')
    parser.add_argument('--thumb-max-velocity',type=positive,default=2.0)
    parser.add_argument('--gain',type=positive,default=1.0)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    if args.headless and not args.video and not (args.max_frames or args.max_seconds):
        parser.error('Headless webcam use requires --max-frames or --max-seconds')
    if not args.hand_model.is_file():
        raise FileNotFoundError(args.hand_model)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    bindings = retarget.joint_bindings(model)
    controller = LiveFilter(model,bindings,args.smoothing_hz,args.thumb_smoothing_hz,
                            args.max_velocity,args.thumb_max_velocity,gain=args.gain)
    thumb = ThumbRetargeter(model,bindings)
    data = mujoco.MjData(model)
    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open {args.video if args.video else "webcam index "+str(args.camera)}; check device/permissions or try --camera 1')
    window = 'Live webcam | Dex5 RIGHT'
    viewer = None
    counts,labels,thumb_counts = Counter(),Counter(),Counter()
    index,previous_ms = 0,-1
    start = time.monotonic()
    try:
        if not args.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if args.video and (not np.isfinite(fps) or fps <= 0):
            raise ValueError('Invalid source video FPS')
        args.output.mkdir(parents=True,exist_ok=False)
        if not args.headless:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(model,data)
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance = max(0.25,model.stat.extent*1.6)
            viewer.cam.azimuth,viewer.cam.elevation = 135,-25
            cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(args.hand_model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=1,
            min_hand_detection_confidence=0.5,min_hand_presence_confidence=0.5,min_tracking_confidence=0.5)
        print('Simulation only. Present ONE physical RIGHT hand; raw labels do not change mapping.',flush=True)
        print('Preview may be mirrored; tracking coordinates are never mirrored. Q/Escape exits.',flush=True)
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as tracker, (args.output/'joint_targets.csv').open('w',newline='') as log:
            writer = csv.writer(log)
            writer.writerow(['frame','source_time_s','state','raw_label','inference_ms','thumb_state','thumb_ik_fit_mm',*bindings])
            while viewer is None or viewer.is_running():
                begin = time.monotonic()
                if args.max_seconds is not None and begin-start >= args.max_seconds:
                    break
                ok,frame = cap.read()
                if not ok:
                    if args.video:
                        break
                    raise RuntimeError('Webcam stopped delivering frames; closing live test')
                source_time = index/fps if args.video else time.monotonic()-start
                ms = max(previous_ms+1,round(source_time*1000))
                previous_ms = ms
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
                inference_start = time.monotonic()
                result = tracker.detect_for_video(image,ms)
                inference_ms = (time.monotonic()-inference_start)*1000
                angles,landmarks,label = read_hand(result)
                world = np.array([[p.x,p.y,p.z] for p in result.hand_world_landmarks[0]]) if angles is not None else None
                thumb_angles,thumb_state,thumb_error = thumb.update(world)
                if angles is not None:
                    angles.update(thumb_angles)
                pose,status = controller.update(angles,source_time)
                retarget.apply_pose(model,data,pose)
                data.time = source_time
                if not np.isfinite(data.xpos).all():
                    raise RuntimeError('Nonfinite MuJoCo kinematics')
                counts[status] += 1
                labels[label] += 1
                thumb_counts[thumb_state] += 1
                writer.writerow([index,source_time,status,label,inference_ms,thumb_state,thumb_error,*[pose[a] for a,_,_ in bindings.values()]])
                if viewer is not None:
                    viewer.sync()
                    cv2.imshow(window,preview_frame(frame,landmarks,status,label,index,not args.no_mirror_preview,inference_ms,thumb_state))
                    if cv2.waitKey(1) & 0xff in (27,ord('q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE) < 1:
                        index += 1
                        break
                    if args.video:
                        time.sleep(max(0,1/fps-(time.monotonic()-begin)))
                index += 1
                if args.max_frames and index >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print('Live test interrupted.',flush=True)
    finally:
        cap.release()
        if viewer is not None:
            viewer.close()
        if not args.headless:
            cv2.destroyAllWindows()
    report = dict(frames=index,source=str(args.video.resolve()) if args.video else f'webcam:{args.camera}',
                  states=dict(counts),raw_labels=dict(labels),physical_hand_assumption='Right',
                  smoothing_hz=args.smoothing_hz,thumb_smoothing_hz=args.thumb_smoothing_hz,
                  max_velocity_rad_s=args.max_velocity,thumb_max_velocity_rad_s=args.thumb_max_velocity,
                  thumb_ik_states=dict(thumb_counts),thumb_method='Four-joint palm-relative bounded IK with persistent-jump gate',
                  missing_policy='Hold up to 0.5s, then return toward neutral with bounded slew; no extrapolation',
                  coordinates_mirrored=False,preview_mirrored=not args.no_mirror_preview,
                  privacy='Local inference; no video/images saved; joint targets and raw labels logged locally',
                  limitations='Other fingers retain 07 flexion mapping; thumb IK approximates palm-relative geometry, no contact/collision validation or occlusion recovery; not hardware control')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Frames: {index}; states: {dict(counts)}; output: {args.output}',flush=True)


if __name__ == '__main__':
    main()
