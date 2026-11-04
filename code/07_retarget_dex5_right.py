"""Replay Test 5 landmarks as a bounded, flexion-only Dex5 right-hand test.

This is kinematic visualization, not motor control or calibrated grasp IK.
Uses hand-centered world landmarks; raw Left/Right labels never mirror input.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT/'outputs/test5_20261029/improved/hand_landmarks.npz'
DEFAULT_MODEL = ROOT/'models/dex5_1/Dex5-URDF-R/Dex5-URDF-R.urdf'
FINGERS = {
    'thumb': ('Yaw_11R', 'Roll_12R', 'Pitch_13R', 'Pitch_14R'),
    'index': ('Roll_21R', 'Pitch_22R', 'Pitch_23R', 'Pitch_24R'),
    'middle': ('Roll_31R', 'Pitch_32R', 'Pitch_33R', 'Pitch_34R'),
    'ring': ('Roll_41R', 'Pitch_42R', 'Pitch_43R', 'Pitch_44R'),
    'pinky': ('Roll_51R', 'Pitch_52R', 'Pitch_53R', 'Pitch_54R'),
}
CHAINS = {'index': (5,6,7,8), 'middle': (9,10,11,12),
          'ring': (13,14,15,16), 'pinky': (17,18,19,20)}


def unit(vector):
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-6:
        raise ValueError('Degenerate landmark bone or palm plane')
    return vector/length


def bend(a, b, c):
    """Unsigned bend: straight segments give zero, a right angle gives pi/2."""
    return float(np.arccos(np.clip(np.dot(unit(b-a), unit(c-b)), -1, 1)))


def landmark_angles(points):
    if points.shape != (21,3) or not np.isfinite(points).all():
        raise ValueError('Expected 21 finite world landmarks')
    normal = unit(np.cross(points[5]-points[0], points[17]-points[0]))
    targets = {}
    for finger, (mcp,pip,dip,tip) in CHAINS.items():
        # Remove lateral spreading from MCP bend estimation in the local palm frame.
        forward = unit(points[mcp]-points[0])
        side = unit(np.cross(normal, forward))
        proximal = unit(points[pip]-points[mcp])
        projected = unit(proximal-side*np.dot(proximal, side))
        mcp_bend = float(np.arccos(np.clip(np.dot(forward, projected), -1, 1)))
        names = FINGERS[finger]
        targets[names[1]] = mcp_bend
        targets[names[2]] = bend(points[mcp], points[pip], points[dip])
        targets[names[3]] = bend(points[pip], points[dip], points[tip])
    # Thumb CMC approximation plus MCP/IP bends. Yaw/opposition stays neutral.
    targets['Roll_12R'] = bend(points[0], points[1], points[2])
    targets['Pitch_13R'] = bend(points[1], points[2], points[3])
    targets['Pitch_14R'] = bend(points[2], points[3], points[4])
    return targets


def joint_bindings(model):
    bindings = {}
    for names in FINGERS.values():
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f'Missing or non-hinge right-hand joint: {name}')
            if not model.jnt_limited[jid]:
                raise ValueError(f'Expected bounded joint: {name}')
            low, high = model.jnt_range[jid]
            bindings[name] = (int(model.jnt_qposadr[jid]), float(low), float(high))
    return bindings


def load_source(path, raw=False):
    with np.load(path, allow_pickle=False) as archive:
        timestamps = np.asarray(archive['timestamps'], dtype=float)
        detected = np.asarray(archive['detected'], dtype=bool)
        interpolated = np.asarray(archive['interpolated'], dtype=bool) if 'interpolated' in archive else np.zeros_like(detected)
        key = 'world_landmarks' if raw else 'processed_world_landmarks'
        if key not in archive:
            raise ValueError(f'Missing {key}; use --raw for original world landmarks')
        points = np.asarray(archive[key], dtype=float)
        labels = np.asarray(archive['handedness'])
        source = np.asarray(archive['source']) if 'source' in archive else np.where(detected, 'observed', 'missing')
        fps = float(archive['fps'])
        assumption = str(archive['assumed_physical_hand']) if 'assumed_physical_hand' in archive else 'Right'
    n = len(timestamps)
    if not n or points.shape != (n,21,3) or any(a.shape != (n,) for a in (detected, interpolated, labels, source)):
        raise ValueError('Input arrays are empty or not frame-aligned')
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0) or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid timestamps/FPS')
    if assumption.lower() != 'right':
        raise ValueError('This test requires a physical right-hand recording; no automatic mirroring')
    eligible = detected | (interpolated if not raw else np.zeros_like(detected))
    eligible &= np.isfinite(points).all(axis=(1,2))
    return points, timestamps, eligible, detected, interpolated, labels, source, fps


def retarget(model, bindings, points, timestamps, eligible, detected, interpolated, gain, smoothing_hz):
    neutral = model.qpos0.copy()
    for address, low, high in bindings.values():
        neutral[address] = np.clip(0, low, high)
    previous = neutral.copy()
    trajectory = np.empty((len(points),model.nq))
    mapped = np.zeros(len(points),dtype=bool)
    states = np.full(len(points), 'missing_hold', dtype='<U24')
    clipped = np.zeros(len(points),dtype=int)
    for i, hand in enumerate(points):
        if eligible[i]:
            try:
                angles = landmark_angles(hand)
            except ValueError:
                states[i] = 'degenerate_hold'
            else:
                target = neutral.copy()
                for name, angle in angles.items():
                    address, low, high = bindings[name]
                    value = gain*angle
                    clipped[i] += int(value < low or value > high)
                    target[address] = np.clip(value, low, high)
                dt = timestamps[i]-timestamps[i-1] if i else 0
                # Initialize the first usable pose directly; smooth later observations only.
                alpha = 1.0 if not mapped[:i].any() or smoothing_hz == 0 else -np.expm1(-2*np.pi*smoothing_hz*dt)
                previous += alpha*(target-previous)
                mapped[i] = True
                states[i] = 'interpolated' if interpolated[i] and not detected[i] else 'observed'
        trajectory[i] = previous
    if not mapped.any():
        raise ValueError('No nondegenerate right-hand world-landmark frames available')
    return trajectory, mapped, states, clipped


def apply_pose(model, data, pose):
    data.qpos[:] = pose
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def playback(model, trajectory, timestamps, mapped, states, fps, speed, full_timeline, loop):
    import mujoco.viewer
    data = mujoco.MjData(model)
    indices = np.flatnonzero(mapped)
    start, end = (0,len(trajectory)-1) if full_timeline else (int(indices[0]),int(indices[-1]))
    apply_pose(model, data, trajectory[start])
    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = max(0.25, model.stat.extent*1.6)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        while viewer.is_running():
            last_print = -1
            for i in range(start,end+1):
                if not viewer.is_running():
                    return
                begin = time.monotonic()
                apply_pose(model,data,trajectory[i])
                data.time = float(timestamps[i])
                viewer.sync()
                if int(timestamps[i]) != last_print:
                    print(f'frame={i} time={timestamps[i]:.3f}s state={states[i]}', flush=True)
                    last_print = int(timestamps[i])
                dt = timestamps[i+1]-timestamps[i] if i < end else 1/fps
                # Never skip poses or replay a backlog in a fast catch-up burst.
                time.sleep(max(0,dt/speed-(time.monotonic()-begin)))
            if not loop:
                print('Playback complete; holding the final pose. Close the viewer to exit.', flush=True)
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(1/30)
                return


def clock_frame(timestamps, elapsed_s, speed=1.0):
    """One shared source-time index for BOTH video and MuJoCo."""
    source_time = timestamps[0]+max(0.0,elapsed_s)*speed
    return int(np.clip(np.searchsorted(timestamps,source_time,side='right')-1,0,len(timestamps)-1))


def comparison_playback(model, trajectory, timestamps, states, fps, speed, loop,
                        video_path, timing_path, exit_at_end=False):
    import cv2
    import mujoco.viewer
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open comparison video: {video_path}')
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isclose(video_fps,fps,rtol=1e-4) or video_frames != len(trajectory):
        cap.release()
        raise ValueError('Comparison video FPS/frame count does not match landmark recording')
    window = 'vid_1 | synchronized Dex5 comparison'
    data = mujoco.MjData(model)
    duration = float(timestamps[-1]-timestamps[0]+1/fps)
    try:
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window,640,720)
        apply_pose(model,data,trajectory[0])
        with timing_path.open('w',newline='') as log, mujoco.viewer.launch_passive(model,data) as viewer:
            writer = csv.writer(log)
            writer.writerow(['cycle','frame','source_time_s','playback_wall_s','presentation_lateness_s','state'])
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance = max(0.25,model.stat.extent*1.6)
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            origin = time.monotonic()
            paused_at, paused_total = None, 0.0
            displayed, decoded, cycle, last_second = -1, -1, 0, -1
            frame = None
            completed = False
            while viewer.is_running():
                now = time.monotonic()
                elapsed = (paused_at if paused_at is not None else now)-origin-paused_total
                index = clock_frame(timestamps,elapsed,speed)
                if index != displayed:
                    # Decode skipped video frames, but only present the shared clock index.
                    while decoded < index:
                        ok,frame = cap.read()
                        if not ok:
                            raise RuntimeError(f'Comparison video ended before frame {index}')
                        decoded += 1
                    apply_pose(model,data,trajectory[index])
                    data.time = float(timestamps[index])
                    preview = frame.copy()
                    h,w = preview.shape[:2]
                    scale = min(640/w,720/h)
                    preview = cv2.resize(preview,(max(1,round(w*scale)),max(1,round(h*scale))))
                    cv2.rectangle(preview,(0,0),(preview.shape[1],75),(0,0,0),-1)
                    cv2.putText(preview,f'Frame {index}/{len(trajectory)-1} | source {timestamps[index]:.3f}s',
                                (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
                    cv2.putText(preview,f'Dex5: {states[index]} | speed {speed:g}x',
                                (10,55),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,220,255),1)
                    cv2.imshow(window,preview)
                    viewer.sync()
                    presented = time.monotonic()-origin-paused_total
                    lateness = presented-(timestamps[index]-timestamps[0])/speed
                    writer.writerow([cycle,index,float(timestamps[index]),presented,lateness,states[index]])
                    displayed = index
                    if int(timestamps[index]) != last_second:
                        print(f'COMPARE frame={index} source={timestamps[index]:.3f}s wall={presented:.3f}s state={states[index]}',flush=True)
                        last_second = int(timestamps[index])
                else:
                    viewer.sync()
                if elapsed*speed >= duration and paused_at is None:
                    if loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                        origin = time.monotonic()
                        paused_total = 0.0
                        displayed,decoded,last_second = -1,-1,-1
                        cycle += 1
                        continue
                    if not completed:
                        print(f'Comparison complete: source duration={duration:.3f}s, wall={elapsed:.3f}s; timing log: {timing_path}',flush=True)
                        completed = True
                    if exit_at_end:
                        return
                key = cv2.waitKey(5) & 0xff
                if key in (27,ord('q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE) < 1:
                    return
                if key == ord(' ') and not completed:
                    if paused_at is None:
                        paused_at = time.monotonic()
                    else:
                        paused_total += time.monotonic()-paused_at
                        paused_at = None
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/datetime.now().strftime('test7_retarget_%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--headless',action='store_true',help='Validate/export without opening a viewer')
    parser.add_argument('--raw',action='store_true',help='Exclude interpolated frames and use raw world landmarks')
    parser.add_argument('--full-timeline',action='store_true',help='Include leading/trailing missing intervals (neutral/hold)')
    parser.add_argument('--loop',action='store_true')
    parser.add_argument('--compare-video',type=Path,nargs='?',const=ROOT/'videos/vid_1.mp4',
                        help='Open source video and MuJoCo on a shared clock; defaults to videos/vid_1.mp4')
    parser.add_argument('--exit-at-end',action='store_true',help='Close comparison windows after one playback')
    parser.add_argument('--speed',type=float,default=1.0)
    parser.add_argument('--gain',type=float,default=1.0,help='Scale bend angles before joint-limit clamping')
    parser.add_argument('--smoothing-hz',type=float,default=5.0,help='Joint-target low-pass cutoff; 0 disables')
    args = parser.parse_args()
    if args.compare_video and args.headless:
        parser.error('--compare-video requires a GUI; omit --headless')
    if args.exit_at_end and not args.compare_video:
        parser.error('--exit-at-end requires --compare-video')
    if any(not np.isfinite(v) or v <= 0 for v in (args.speed,args.gain)) or not np.isfinite(args.smoothing_hz) or args.smoothing_hz < 0:
        parser.error('Speed/gain must be positive and finite; smoothing cutoff must be nonnegative and finite')
    points,timestamps,eligible,detected,interpolated,labels,source,fps = load_source(args.data,args.raw)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    bindings = joint_bindings(model)
    trajectory,mapped,states,clipped = retarget(model,bindings,points,timestamps,eligible,detected,interpolated,args.gain,args.smoothing_hz)
    validation_data = mujoco.MjData(model)
    for pose in trajectory:
        apply_pose(model,validation_data,pose)
        if not np.isfinite(validation_data.xpos).all():
            raise RuntimeError('MuJoCo forward kinematics produced nonfinite positions')
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'dex5_joint_targets.npz', qpos=trajectory,timestamps=timestamps,
                        mapped=mapped,state=states,source=source,detected=detected,interpolated=interpolated,
                        raw_handedness=labels,assumed_physical_hand=np.array('Right'),
                        joint_names=np.array(list(bindings)),qpos_addresses=np.array([x[0] for x in bindings.values()]))
    with (args.output/'joint_targets.csv').open('w',newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['frame','time_s','state','source','raw_handedness',*bindings])
        for i,pose in enumerate(trajectory):
            writer.writerow([i,timestamps[i],states[i],source[i],labels[i],*[pose[x[0]] for x in bindings.values()]])
    report = dict(input=str(args.data.resolve()),model=str(args.model.resolve()),frames=len(points),
                  mapped_frames=int(mapped.sum()), observed_mapped_frames=int((mapped & detected).sum()),
                  interpolated_mapped_frames=int((mapped & ~detected & interpolated).sum()),
                  held_frames=int((~mapped).sum()),clamped_joint_values=int(clipped.sum()),
                  gain=args.gain,smoothing_hz=args.smoothing_hz,raw_only=args.raw,
                  joint_ranges_radians={n:[low,high] for n,(_,low,high) in bindings.items()},
                  max_joint_excursion_radians={n:float(np.ptp(trajectory[:,a])) for n,(a,_,_) in bindings.items()},
                  method='Unsigned landmark bend-angle approximation, 15 flexion joints; 5 spread/yaw joints neutral',
                  missing_policy='Neutral before first valid frame, then hold last pose; held frames are never observations',
                  limitations='No wrist motion, calibrated thumb opposition, collision avoidance, IK, torque or hardware control; tracking outliers can remain',
                  actuators=int(model.nu),mujoco_forward_checked_frames=len(trajectory))
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'Dex5 RIGHT: {model.nq} qpos, {model.nu} actuators; kinematic visualization only.')
    print(f'Mapped {mapped.sum()}/{len(mapped)} frames; observed={report["observed_mapped_frames"]}, interpolated={report["interpolated_mapped_frames"]}, held={report["held_frames"]}')
    print(f'Exported: {args.output}',flush=True)
    if not args.headless:
        if args.compare_video:
            comparison_playback(model,trajectory,timestamps,states,fps,args.speed,args.loop,
                                args.compare_video,args.output/'comparison_timing.csv',args.exit_at_end)
        else:
            playback(model,trajectory,timestamps,mapped,states,fps,args.speed,args.full_timeline,args.loop)


if __name__ == '__main__':
    main()
