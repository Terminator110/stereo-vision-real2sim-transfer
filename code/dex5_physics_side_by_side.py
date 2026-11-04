#!/usr/bin/env python3
"""Create camera1 vs MuJoCo physics replay MP4 for a DEX5 recorded take.

Video stays on the last recorded grasp frame during simulated hold and lift.
The physics replay is *not* modified; this visualizes saved physics_replay.npz.
"""
import argparse
import csv
import os
from pathlib import Path

# Set before importing mujoco; offscreen EGL works on Ubuntu NVIDIA systems.
os.environ.setdefault('MUJOCO_GL', 'egl')

import cv2
import mujoco
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--take', default='take_20261104_175536_951802')
    ap.add_argument('--root', type=Path, default=Path.home()/'physikal_workflow/dex-5')
    ap.add_argument('--physics-dir', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--grasp-frame', type=int, default=85)
    ap.add_argument('--fps', type=float, default=15)
    ap.add_argument('--panel-width', type=int, default=800)
    ap.add_argument('--panel-height', type=int, default=600)
    ap.add_argument('--camera-azimuth', type=float, default=130)
    ap.add_argument('--camera-elevation', type=float, default=-20)
    ap.add_argument('--camera-distance', type=float, default=.9)
    ap.add_argument('--settle-seconds', type=float, default=.4)
    args=ap.parse_args()
    root=args.root.expanduser().resolve()
    physics=args.physics_dir.expanduser().resolve()
    dest=args.output.expanduser().resolve()
    scene=physics/'scenario_dynamic_replay.xml'
    replay=physics/'physics_replay.npz'
    trace=physics/'physics_trace.csv'
    recorded=root/'outputs'/f'smoothed_{args.take}'/'sim_qpos.npz'
    video=root/'code/stereo/recordings_three_cam'/args.take/'camera1.avi'
    for p in (scene,replay,trace,recorded,video):
        if not p.is_file():
            raise FileNotFoundError(p)
    if args.fps<=0 or args.panel_height<160 or args.panel_width<160:
        raise ValueError('Invalid fps/panel size')

    with np.load(replay) as f:
        sim_qpos=np.array(f['qpos']);sim_times=np.array(f['time_s'])
    with np.load(recorded) as f:
        recorded_times=np.array(f['time_s']);recorded_frames=np.array(f['source_frames'])
    if len(sim_qpos)==0 or len(sim_qpos)!=len(sim_times):
        raise ValueError('Invalid physics replay arrays')
    if len(recorded_times)==0 or len(recorded_times)!=len(recorded_frames):
        raise ValueError('Invalid recorded replay arrays')

    with trace.open(newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=len(sim_times):
        raise ValueError(f'physics_trace ({len(rows)}) and replay ({len(sim_times)}) differ')
    phases=[r['phase'] for r in rows]

    matches=np.where(recorded_frames==args.grasp_frame)[0]
    if len(matches)!=1:
        raise ValueError(f'Grasp frame {args.grasp_frame} not present in recorded frames')
    grasp_i=int(matches[0]); recorded_t=recorded_times-recorded_times[0]
    grasp_time=float(recorded_t[grasp_i])
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera video: {video}')
    model=mujoco.MjModel.from_xml_path(str(scene))
    data=mujoco.MjData(model)
    if model.nq!=sim_qpos.shape[1]:
        raise ValueError(f'Physics replay qpos has {sim_qpos.shape[1]} columns, XML needs {model.nq}')

    data.qpos[:]=sim_qpos[0];mujoco.mj_forward(model,data)
    cupid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'table_cup')
    sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'g1_01_right_wrist_ik_site')
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    if cupid>=0:
        camera.lookat[:]=data.xpos[cupid]+np.array([0,0,.14])
    elif sid>=0:
        camera.lookat[:]=data.site_xpos[sid]
    camera.distance=args.camera_distance
    camera.azimuth=args.camera_azimuth
    camera.elevation=args.camera_elevation

    w,h=args.panel_width,args.panel_height
    dest.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(2*w,h))
    if not writer.isOpened():
        raise RuntimeError('Could not create MP4; check OpenCV video codec support')

    sample_count=int(np.ceil(float(sim_times[-1])*args.fps))+1
    print('Source camera:',video,flush=True)
    print('MuJoCo scene:',scene,flush=True)
    print('Video frames:',sample_count,'at',args.fps,'fps',flush=True)
    try:
        with mujoco.Renderer(model,height=h,width=w) as renderer:
            for fi in range(sample_count):
                t=min(fi/args.fps,float(sim_times[-1]))
                si=int(np.clip(np.searchsorted(sim_times,t,side='left'),0,len(sim_times)-1))
                phase=phases[si]
                # The original camera has no commanded lift: freeze at grasp frame.
                original_elapsed=max(0.,t-args.settle_seconds)
                original_elapsed=min(original_elapsed,grasp_time)
                ri=int(np.clip(np.searchsorted(recorded_t,original_elapsed,side='left'),0,grasp_i))
                vf=int(recorded_frames[ri])
                cap.set(cv2.CAP_PROP_POS_FRAMES,vf)
                ok,frame=cap.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode camera1 frame {vf} at simulated t={t:.3f}s')
                frame=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)
                data.qpos[:]=sim_qpos[si]
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera)
                rgb=renderer.render()
                sim=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                combined=np.hstack((frame,sim))
                # Text labels and status bar
                cv2.rectangle(combined,(0,0),(2*w,45),(25,25,25),-1)
                cv2.putText(combined,f'ORIGINAL CAMERA  |  frame {vf}',(14,30),cv2.FONT_HERSHEY_SIMPLEX,.66,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(combined,f'MUJOCO PHYSICS  |  {phase}  |  t={t:.2f}s',(w+14,30),cv2.FONT_HERSHEY_SIMPLEX,.60,(255,255,255),2,cv2.LINE_AA)
                if phase in ('grasp_hold','lift','hold'):
                    cv2.putText(combined,'Original video held at grasp frame (no recorded lift)',(14,h-18),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,240,255),1,cv2.LINE_AA)
                writer.write(combined)
                if fi%max(1,int(args.fps*3))==0:
                    print(f'Encoded {fi}/{sample_count}, phase={phase}',flush=True)
    finally:
        writer.release();cap.release()
    if not dest.is_file() or dest.stat().st_size<1000:
        raise RuntimeError('MP4 output was not written correctly')
    print('Saved:',dest, 'bytes:',dest.stat().st_size,flush=True)


if __name__=='__main__':
    main()
