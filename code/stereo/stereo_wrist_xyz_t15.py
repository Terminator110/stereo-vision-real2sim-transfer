#!/usr/bin/env python3
"""T15 independent wrist XYZ stereo diagnostic; no wrist orientation estimate.

Raw 3D position is MediaPipe landmark 0, triangulated using K1,D1,K2,D2,R,T.
Palm-centre proxy is measured separately from MCP landmarks; it is NOT the wrist.
No model-generated positions, extrapolation, or monocular 3D coordinates are emitted.
"""
import argparse
import csv
import os
import time
import cv2
import numpy as np
import mediapipe as mp

MCP = (5, 9, 13, 17)


def calibration(path):
    with np.load(path, allow_pickle=False) as f:
        data = dict(f)
    def get(*names):
        for name in names:
            if name in data:
                return np.asarray(data[name], dtype=np.float64)
        raise ValueError(f'Missing calibration key: {names}; available={list(data)}')
    return (get('K1', 'cameraMatrix1').reshape(3,3), get('D1', 'distCoeffs1').reshape(-1,1),
            get('K2', 'cameraMatrix2').reshape(3,3), get('D2', 'distCoeffs2').reshape(-1,1),
            get('R', 'R_stereo').reshape(3,3), get('T', 'T_stereo').reshape(3,1))


def detector(model):
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1, min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4, min_tracking_confidence=0.4)
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def infer(det, frame, stamp):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp)
    if not result.hand_landmarks:
        return None
    h, w = frame.shape[:2]
    return np.asarray([(p.x * w, p.y * h) for p in result.hand_landmarks[0]], np.float64)


def project(xyz, K, D, R, T):
    rv, _ = cv2.Rodrigues(R)
    xy, _ = cv2.projectPoints(np.asarray(xyz, dtype=np.float64).reshape(-1,3), rv, T, K, D)
    return xy.reshape(-1,2)


def stereo(p0, p1, c):
    K0,D0,K1,D1,R,T = c
    a = cv2.undistortPoints(p0.reshape(-1,1,2), K0,D0).reshape(-1,2)
    b = cv2.undistortPoints(p1.reshape(-1,1,2), K1,D1).reshape(-1,2)
    P0 = np.column_stack((np.eye(3), np.zeros(3)))
    P1 = np.column_stack((R,T))
    hom = cv2.triangulatePoints(P0,P1,a.T,b.T)
    xyz = np.full((len(p0),3), np.nan)
    goodw = np.abs(hom[3]) > 1e-12
    xyz[goodw] = (hom[:3,goodw]/hom[3,goodw]).T
    err = np.full(len(p0), np.nan)
    fin = np.isfinite(xyz).all(axis=1)
    if fin.any():
        left = project(xyz[fin],K0,D0,np.eye(3),np.zeros((3,1)))
        right = project(xyz[fin],K1,D1,R,T)
        err[fin] = np.maximum(np.linalg.norm(left-p0[fin],axis=1), np.linalg.norm(right-p1[fin],axis=1))
    zright = (R@xyz.T+T).T[:,2]
    front = fin & (xyz[:,2]>0) & (zright>0)
    return xyz,err,front


def open_cam(source,w,h):
    try:
        source = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {source}')
    if isinstance(source,int):
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
        cap.set(cv2.CAP_PROP_FPS,30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    got = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (w,h):
        raise RuntimeError(f'Camera {source}: got {got}, expected {(w,h)}')
    print(f'Camera {source} {got}')
    return cap


def fmt(v):
    return '' if not np.isfinite(v) else f'{v:.6f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calib',default='calibration/results/stereo_calibration.npz')
    p.add_argument('--model',default='hand_landmarker.task')
    p.add_argument('--left',default='2'); p.add_argument('--right',default='0')
    p.add_argument('--width',type=int,default=1280); p.add_argument('--height',type=int,default=720)
    p.add_argument('--max-reproj',type=float,default=5.0)
    p.add_argument('--min-mcp',type=int,default=3)
    p.add_argument('--output',default='tests/T15_wrist_xyz.csv')
    p.add_argument('--no-display',action='store_true')
    args = p.parse_args()
    if not os.path.isfile(args.model):
        p.error(f'Missing model {args.model}')
    c = calibration(args.calib)
    print(f'Stereo T baseline={np.linalg.norm(c[5]):.5f} calibration units; XYZ in same units')
    a = open_cam(args.left,args.width,args.height)
    b = open_cam(args.right,args.width,args.height)
    dl = detector(args.model); dr = detector(args.model)
    columns = ['frame','time_s','mode','reason','left_detected','right_detected',
               'x','y','z','wrist_reproj_px','wrist_front','valid_mcp_count',
               'palm_x','palm_y','palm_z','palm_wrist_distance',
               'mcp_reproj_median_px','left_wrist_u','left_wrist_v','right_wrist_u','right_wrist_v']
    os.makedirs(os.path.dirname(args.output) or '.',exist_ok=True)
    start=time.monotonic(); frame=0
    try:
        with open(args.output,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader()
            while a.grab() and b.grab():
                oka,iml = a.retrieve(); okb,imr = b.retrieve()
                if not (oka and okb): break
                now=time.monotonic(); t=now-start; stamp=max(1,int(t*1000))
                pl=infer(dl,iml,stamp); pr=infer(dr,imr,stamp)
                row={key:'' for key in columns}
                row.update(frame=frame,time_s=f'{t:.5f}',left_detected=int(pl is not None),right_detected=int(pr is not None),mode='INVALID',reason='neither_detected')
                if pl is None and pr is not None: row['reason']='left_not_detected'
                elif pl is not None and pr is None: row['reason']='right_not_detected'
                elif pl is not None and pr is not None:
                    row.update(left_wrist_u=fmt(pl[0,0]),left_wrist_v=fmt(pl[0,1]),right_wrist_u=fmt(pr[0,0]),right_wrist_v=fmt(pr[0,1]))
                    xyz,errors,front=stereo(pl,pr,c)
                    row['wrist_reproj_px']=fmt(errors[0]); row['wrist_front']=int(front[0])
                    valid = front & np.isfinite(errors) & (errors <= args.max_reproj)
                    mcps=[i for i in MCP if valid[i]]
                    row['valid_mcp_count']=len(mcps)
                    if mcps:
                        row['mcp_reproj_median_px']=fmt(np.median(errors[mcps]))
                    if len(mcps)>=args.min_mcp:
                        palm=xyz[mcps].mean(axis=0)
                        row.update(palm_x=fmt(palm[0]),palm_y=fmt(palm[1]),palm_z=fmt(palm[2]))
                        if front[0]:
                            row['palm_wrist_distance']=fmt(np.linalg.norm(palm-xyz[0]))
                    if not front[0]: row['reason']='wrist_behind_camera'
                    elif not np.isfinite(errors[0]): row['reason']='wrist_nonfinite_error'
                    elif errors[0]>args.max_reproj: row['reason']='wrist_reprojection_high'
                    else:
                        row.update(mode='STEREO_WRIST',reason='ok',x=fmt(xyz[0,0]),y=fmt(xyz[0,1]),z=fmt(xyz[0,2]))
                writer.writerow(row)
                if frame % 30 == 0:
                    f.flush()
                    print(f'{frame:05d} {row["mode"]:13} {row["reason"]:27} wrist_err={row["wrist_reproj_px"]} mcp={row["valid_mcp_count"]}')
                if not args.no_display:
                    for image,points,name in ((iml,pl,'LEFT'),(imr,pr,'RIGHT')):
                        if points is not None:
                            for j in (0,*MCP):
                                cv2.circle(image,tuple(np.rint(points[j]).astype(int)),5,(0,0,255) if j==0 else (0,255,0),-1)
                        cv2.putText(image,f'{name} {row["mode"]}  {row["reason"]}',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,0),2)
                    pair=np.hstack((iml,imr)) # No flipping: display is true raw orientation.
                    if pair.shape[1]>1800:
                        scale=1800/pair.shape[1]; pair=cv2.resize(pair,None,fx=scale,fy=scale)
                    cv2.imshow('T15 position only (q quit)',pair)
                    if cv2.waitKey(1)&255==ord('q'):
                        frame+=1; break
                frame+=1
    finally:
        dl.close();dr.close();a.release();b.release();cv2.destroyAllWindows()
        print(f'Saved {frame} frames to {args.output}')

if __name__=='__main__':main()
