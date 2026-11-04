#!/usr/bin/env python3
"""Create 4-panel MP4: source camera, stereo palm frame, IK target, actual Dex5 site frame.

All rotations are displayed in SAME MuJoCo world basis. Target reconstructs the
TEMPORARY first-frame alignment used in test_g1_palm_orientation_ik.py.
No cameras, inference, or simulation stepping required.
"""
import argparse
from pathlib import Path
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def unit(v):
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1.e-8:
        raise ValueError('degenerate palm')
    return v / n


def basis(p):
    wrist, idx, mid, pinky = p[[0, 5, 9, 17]]
    forward = unit(mid - wrist)
    across = unit((idx - pinky) - forward * np.dot(idx - pinky, forward))
    normal = unit(np.cross(across, forward))
    return np.column_stack((across, forward, normal))


def angle(A, B):
    return float(np.degrees(Rotation.from_matrix(A @ B.T).magnitude()))


def panel(title, R, w=510, h=360, footer='', meanings=None):
    pic = np.full((h, w, 3), (28, 30, 34), np.uint8)
    cv2.putText(pic, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, .68, (245,245,245), 2, cv2.LINE_AA)
    if footer:
        cv2.putText(pic, footer, (16, h-16), cv2.FONT_HERSHEY_SIMPLEX, .47, (200,200,200), 1, cv2.LINE_AA)
    if R is None:
        cv2.putText(pic, 'NO VALID STEREO PALM', (48, h//2), cv2.FONT_HERSHEY_SIMPLEX, .7, (100,180,255), 2)
        return pic
    # Fixed oblique virtual camera looking toward origin, same projection for all panels.
    az, el = np.deg2rad(45), np.deg2rad(25)
    eye = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    right = unit(np.cross(np.array([0.,0.,1.]), eye))
    up = unit(np.cross(eye, right))
    def proj(v):
        return (int(w*.5 + 115*np.dot(v,right)), int(h*.55 - 115*np.dot(v,up)))
    origin = proj(np.zeros(3))
    cv2.circle(pic, origin, 5, (235,235,235), -1)
    axes = [('X', (60,80,250)), ('Y', (60,220,70)), ('Z', (255,155,65))]
    for j,(name,color) in enumerate(axes):
        pt = proj(R[:,j]); cv2.arrowedLine(pic, origin, pt, color, 5, cv2.LINE_AA, tipLength=.17)
        cv2.putText(pic, name, (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    cv2.putText(pic, 'RED +X      GREEN +Y      BLUE +Z', (16,55), cv2.FONT_HERSHEY_SIMPLEX, .48, (210,210,210), 1)
    if meanings:
        for line_no,line in enumerate(meanings):
            cv2.putText(pic, line, (14, 79+line_no*21), cv2.FONT_HERSHEY_SIMPLEX, .43, (220,220,220), 1, cv2.LINE_AA)
    return pic


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--take',type=Path,required=True)
    ap.add_argument('--model',type=Path,default=Path('../models/scenario_g1_dex5_stationary.xml'))
    ap.add_argument('--start',type=int,default=102)
    ap.add_argument('--end',type=int,default=128)
    ap.add_argument('--source-start',type=int,default=90)
    ap.add_argument('--yaw-deg',type=float,default=0)
    ap.add_argument('--fps',type=float,default=5)
    ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    if a.end<a.start or a.fps<=0: ap.error('invalid start/end/fps')
    take=a.take.resolve(); base=take/'grasp_stereo_dex5_test'
    source=base/'script12_output/stereo_landmarks_and_targets.npz'
    combined=base/'combined_orientation_fingers/combined_qpos.npz'
    for p in (source,combined,take/'camera1.avi'):
        if not p.is_file(): ap.error(f'Missing {p}')
    with np.load(source,allow_pickle=False) as z:
        points=np.asarray(z['points_table_m'],float)
        valid=np.asarray(z['landmark_valid'],bool)
    with np.load(combined,allow_pickle=False) as z:
        frames=np.asarray(z['source_frames'],int)
        qposes=np.asarray(z['qpos'],float)
    if len(frames)!=len(qposes): ap.error('combined frames/qpos length mismatch')
    model=mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    data=mujoco.MjData(model)
    sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'g1_01_dex5_right_frame')
    if sid<0: ap.error('Dex5 right site missing')
    frame_to_q={int(f):q for f,q in zip(frames,qposes)}
    yaw=Rotation.from_euler('z',a.yaw_deg,degrees=True).as_matrix()
    human={}
    for frame in range(a.start,a.end+1):
        i=frame-a.source_start
        if 0<=i<len(points) and valid[i,[0,5,9,17]].all():
            try:human[frame]=yaw @ basis(points[i])
            except ValueError:pass
    if not human:ap.error('No valid palm frames')
    first=min(human)
    if first not in frame_to_q:ap.error(f'first valid frame {first} not in combined qpos')
    # This starting orientation must match the ORIGINAL pre-IK neutral seed used
    # by test_g1_palm_orientation_ik.py, not the already-solved frame's pose.
    data.qpos[:]=model.qpos0
    for name,v in [('g1_01_right_shoulder_roll_joint',-.25),('g1_01_right_elbow_joint',.6)]:
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if jid>=0:data.qpos[model.jnt_qposadr[jid]]=np.clip(v,*model.jnt_range[jid])
    mujoco.mj_forward(model,data)
    robot_ref=data.site_xmat[sid].reshape(3,3).copy()
    human_ref=human[first]
    cam=cv2.VideoCapture(str(take/'camera1.avi'))
    if not cam.isOpened():ap.error('Cannot open camera1.avi')
    output=a.output or base/'palm_frame_comparison.mp4'
    output.parent.mkdir(parents=True,exist_ok=True)
    w,h=1020,720
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    writer=cv2.VideoWriter(str(output),fourcc,a.fps,(w,h))
    if not writer.isOpened():ap.error('Cannot open video writer')
    print(f'Using first stereo valid frame {first}; yaw={a.yaw_deg} degrees',flush=True)
    print('Source: camera1; axes: stereo palm, temporary IK site target, actual G1 site',flush=True)
    print('NOTE: coordinate axes of robot site are NOT necessarily anatomical palm axes.',flush=True)
    try:
        for frame in range(a.start,a.end+1):
            if frame not in frame_to_q: print(f'Skip frame {frame}: no G1 pose');continue
            cam.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,raw=cam.read()
            video=np.full((360,510,3),(35,35,35),np.uint8)
            if ok:
                raw=cv2.resize(raw,(510,287))
                video[65:352,:]=raw
            cv2.putText(video,f'CAMERA 1 | SOURCE FRAME {frame}',(14,30),cv2.FONT_HERSHEY_SIMPLEX,.63,(255,255,255),2)
            cv2.putText(video,'Near-cup grasp; minor finger closing expected', (14,51),cv2.FONT_HERSHEY_SIMPLEX,.45,(200,200,200),1)
            H=human.get(frame)
            # Formula copied from orientation IK test: target=human@human_ref.T@robot_ref
            target=H @ human_ref.T @ robot_ref if H is not None else None
            data.qpos[:]=frame_to_q[frame];mujoco.mj_forward(model,data)
            actual=data.site_xmat[sid].reshape(3,3).copy()
            human_delta = angle(human_ref,H) if H is not None else None
            rawp=panel('HUMAN PALM | stereo 3D',H,
                       footer=(f'Palm rotation since frame {first}: {human_delta:.1f} deg' if human_delta is not None else 'No stereo palm for this frame'),
                       meanings=['+X: pinky -> index across palm',
                                 '+Y: wrist -> middle knuckle',
                                 '+Z: X cross Y (palm normal)',
                                 'Frame axes are in table coordinates + yaw'])
            tpanel=panel('G1 IK TARGET | Dex5 SITE axes',target,
                         footer=f'Frame {first} palm = initial robot hand site',
                         meanings=['+X / +Y / +Z: robot site local axes',
                                   'NOT yet mapped to anatomical palm axes',
                                   'Computed from human relative rotation'])
            apanel=panel('ACTUAL G1 | Dex5 SITE axes',actual,
                         footer=f'IK target error: {angle(target,actual):.2f} deg' if target is not None else 'No target for this frame',
                         meanings=['+X / +Y / +Z: robot site local axes',
                                   'Compare arrow colors with IK TARGET',
                                   'Do NOT compare directly to human axes'])
            if H is not None:
                print(f'{frame}: palm change since {first} = {human_delta:.2f} deg | IK target vs actual = {angle(target,actual):.2f} deg',flush=True)
            writer.write(np.vstack([np.hstack([video,rawp]),np.hstack([tpanel,apanel])]))
    finally:
        writer.release();cam.release()
    print(f'Saved: {output}',flush=True)

if __name__=='__main__':main()
