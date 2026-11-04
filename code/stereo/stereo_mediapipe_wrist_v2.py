#!/usr/bin/env python3
"""Stereo markerless wrist position tracker (MediaPipe Tasks + OpenCV).

Coordinates: camera-0 optical frame, in units of stereo translation T.
Stereo calibration: NPZ/YAML containing K1,D1,K2,D2,R,T (see --help).
Only translation is tracked; rotation is deliberately excluded.
"""
import argparse
import csv
import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# Landmark index 0 = wrist, 5/9/13/17 = metacarpophalangeal knuckles.
TRACK_IDS = (0, 5, 9, 13, 17)
EDGES = [(0,5),(5,9),(9,13),(13,17),(0,17)]


def load_calibration(path):
    if path.endswith('.npz'):
        with np.load(path) as f:
            data = {k: f[k] for k in f.files}
    else:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f'Cannot open calibration: {path}')
        data = {}
        for name in ('K1','D1','K2','D2','R','T','cameraMatrix1','cameraMatrix2',
                     'distCoeffs1','distCoeffs2','R_stereo','T_stereo'):
            node = fs.getNode(name)
            if not node.empty():
                data[name] = node.mat()
        fs.release()
    def pick(*keys):
        for k in keys:
            if k in data:
                return np.asarray(data[k], dtype=np.float64)
        raise ValueError(f'Missing calibration key {keys}; found {list(data)}')
    k1 = pick('K1','cameraMatrix1').reshape(3,3)
    k2 = pick('K2','cameraMatrix2').reshape(3,3)
    d1 = pick('D1','distCoeffs1').reshape(-1,1)
    d2 = pick('D2','distCoeffs2').reshape(-1,1)
    r = pick('R','R_stereo').reshape(3,3)
    t = pick('T','T_stereo').reshape(3,1)
    if not np.isfinite(t).all() or np.linalg.norm(t) < 1e-9:
        raise ValueError('Invalid stereo baseline T')
    if abs(np.linalg.det(r) - 1) > 0.05:
        raise ValueError('Invalid stereo rotation R')
    return k1,d1,k2,d2,r,t


def create_detector(model):
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.40,
        min_hand_presence_confidence=0.40,
        min_tracking_confidence=0.40)
    return mp.tasks.vision.HandLandmarker.create_from_options(opts)


def infer(detector, frame, timestamp_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
    if not result.hand_landmarks:
        return None
    h,w = frame.shape[:2]
    pts = np.asarray([(p.x*w, p.y*h) for p in result.hand_landmarks[0]], np.float64)
    return pts


def project(x, k, d, r=None, t=None):
    if r is None:
        r = np.eye(3)
        t = np.zeros((3,1))
    rv,_ = cv2.Rodrigues(r)
    pts,_ = cv2.projectPoints(np.asarray(x,dtype=np.float64).reshape(-1,3),rv,
                               np.asarray(t,dtype=np.float64).reshape(3,1),k,d)
    return pts.reshape(-1,2)


def triangulate(p1, p2, calib):
    k1,d1,k2,d2,r,t = calib
    a = cv2.undistortPoints(np.asarray(p1,dtype=np.float64).reshape(-1,1,2),k1,d1).reshape(-1,2)
    b = cv2.undistortPoints(np.asarray(p2,dtype=np.float64).reshape(-1,1,2),k2,d2).reshape(-1,2)
    pmat1 = np.column_stack((np.eye(3), np.zeros(3)))
    pmat2 = np.column_stack((r,t))
    hom = cv2.triangulatePoints(pmat1, pmat2, a.T, b.T)
    validw = np.abs(hom[3]) > 1e-12
    xyz = np.full((len(p1),3),np.nan)
    xyz[validw] = (hom[:3,validw]/hom[3,validw]).T
    z2 = (r @ xyz.T + t).T[:,2]
    err1 = np.linalg.norm(project(xyz,k1,d1)-p1,axis=1)
    err2 = np.linalg.norm(project(xyz,k2,d2,r,t)-p2,axis=1)
    return xyz, np.maximum(err1,err2), (xyz[:,2]>0)&(z2>0)&validw


def mono_estimate(template, pts, k, d, r, t, max_error):
    """Estimate camera-0 wrist from a short-lived stereo hand template using PnP.
    Template points are in camera-0 coordinates from one valid stereo observation.
    PnP allows camera/hand movement, but does NOT infer changing hand anatomy.
    """
    indices = [i for i in TRACK_IDS if np.isfinite(template[i]).all()
               and np.isfinite(pts[i]).all()]
    if len(indices) < 4:
        return None, np.nan
    obj = np.asarray([template[i] for i in indices],np.float64)
    img = np.asarray([pts[i] for i in indices],np.float64)
    # Convert template coordinates (cam0) to this camera's image frame.
    # SolvePnP returns object-to-current-camera pose, not camera0 coordinates.
    ok,rv,tv = cv2.solvePnP(obj,img,k,d,flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None,np.nan
    ok,rv,tv = cv2.solvePnP(obj,img,k,d,rv,tv,True,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None,np.nan
    projected,_ = cv2.projectPoints(obj,rv,tv,k,d)
    error = float(np.median(np.linalg.norm(projected.reshape(-1,2)-img,axis=1)))
    rr,_ = cv2.Rodrigues(rv)
    # Wrist template transformed into observed camera coordinate system.
    wrist_camera = (rr @ template[0].reshape(3,1) + tv).reshape(3,1)
    if wrist_camera[2,0] <= 0 or error > max_error:
        return None,error
    # camera-1 X = R * camera-0 X + T  => inverse back to camera-0
    wrist0 = (r.T @ (wrist_camera-t)).reshape(3)
    if wrist0[2] <= 0:
        return None,error
    return wrist0,error


def draw(frame, pts, text, color):
    if pts is not None:
        for i,j in EDGES:
            cv2.line(frame,tuple(np.rint(pts[i]).astype(int)),
                     tuple(np.rint(pts[j]).astype(int)),color,2)
        for idx in TRACK_IDS:
            cv2.circle(frame,tuple(np.rint(pts[idx]).astype(int)),5,
                       (0,0,255) if idx==0 else color,-1)
    cv2.putText(frame,text,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2)
    return frame


def open_source(source, width, height):
    try:
        source=int(source)
    except ValueError:
        pass
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera/video {source}')
    if isinstance(source,int):
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        cap.set(cv2.CAP_PROP_FPS,30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        actual=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f'Camera {source}: {actual}, requested {(width,height)}')
        if actual!=(width,height):
            raise RuntimeError('Camera size differs from calibration: '+str(actual))
    return cap


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calib',required=True,help='Stereo .npz/.yml with K1,D1,K2,D2,R,T; T units set output units')
    ap.add_argument('--model',default='hand_landmarker.task',help='MediaPipe Hand Landmarker model')
    ap.add_argument('--left',default='0',help='Camera index or video path (camera 0 reference)')
    ap.add_argument('--right',default='1',help='Camera index or video path (camera 1)')
    ap.add_argument('--width',type=int,default=1280)
    ap.add_argument('--height',type=int,default=720)
    ap.add_argument('--allow-mono',action='store_true',help='Enable experimental monocular hand PnP fallback')
    ap.add_argument('--max-reproj',type=float,default=5.,help='Maximum reprojection error in original image pixels')
    ap.add_argument('--max-mono-reproj',type=float,default=7.)
    ap.add_argument('--template-age',type=float,default=0.75,help='Max age of stereo hand geometry for monocular PnP, seconds')
    ap.add_argument('--max-gap',type=float,default=0.35,help='Maximum EMA continuity gap, seconds')
    ap.add_argument('--alpha',type=float,default=0.55,help='EMA filter alpha; 1 = no smoothing')
    ap.add_argument('--output',default='wrist_markerless_3d.csv')
    ap.add_argument('--no-display',action='store_true')
    args=ap.parse_args()
    if not os.path.isfile(args.model):
        ap.error(f'Model not found: {args.model}')
    calib=load_calibration(args.calib)
    k1,d1,k2,d2,r,t=calib
    baseline=np.linalg.norm(t)
    print(f'Stereo baseline: {baseline:.3f} calibration units. Output uses SAME units.')
    print('IMPORTANT: input videos/cameras MUST match calibration resolution and orientation.')
    left=open_source(args.left,args.width,args.height)
    right=open_source(args.right,args.width,args.height)
    det0=create_detector(args.model)
    det1=create_detector(args.model)
    template=None
    template_at=-np.inf
    smooth=None
    smooth_at=-np.inf
    last_mode='NONE'
    last_time=time.monotonic()
    frame_id=0
    start=time.monotonic()
    cols=['frame','time_s','mode','left_detected','right_detected',
          'x','y','z','smooth_x','smooth_y','smooth_z','reproj_px','pnp_px',
          'wrist_reproj_px','wrist_front','knuckles_good','reason']
    try:
        with open(args.output,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=cols)
            writer.writeheader()
            while True:
                # grab both before retrieve to minimize (NOT eliminate) frame timing skew.
                if not left.grab() or not right.grab():
                    break
                ok0,a=left.retrieve()
                ok1,b=right.retrieve()
                if not (ok0 and ok1):
                    break
                now=time.monotonic()
                elapsed=now-start
                ts=max(1,int(elapsed*1000))
                # The cameras have separate task instances, each with increasing timestamps.
                p0=infer(det0,a,ts)
                p1=infer(det1,b,ts)
                wrist=None
                reproj=np.nan
                pnp_err=np.nan
                mode='INVALID'
                reason='neither_detected'
                wrist_err=np.nan
                wrist_front=0
                knuckles_good=0
                if p0 is None and p1 is not None: reason='left_not_detected'
                if p0 is not None and p1 is None: reason='right_not_detected'
                if p0 is not None and p1 is not None:
                    xyz,errs,front=triangulate(p0,p1,calib)
                    subset=np.asarray(TRACK_IDS)
                    good=front[subset] & (errs[subset]<=args.max_reproj)
                    # Wrist position is independent of knuckle validity.
                    wrist_err=float(errs[0]) if np.isfinite(errs[0]) else np.nan
                    wrist_front=int(bool(front[0]))
                    knuckles_good=int(np.count_nonzero(good[1:]))
                    if not front[0]:
                        reason='wrist_behind_camera'
                    elif not np.isfinite(wrist_err):
                        reason='wrist_nonfinite_error'
                    elif wrist_err>args.max_reproj:
                        reason='wrist_reprojection_high'
                    else:
                        wrist=xyz[0].copy()
                        reproj=wrist_err
                        mode='STEREO'
                        reason='ok'
                        # Only complete, coherent stereo hand shapes form PnP templates.
                        if np.all(good):
                            template=xyz.copy()
                            template_at=now
                if (args.allow_mono and wrist is None and template is not None
                        and now-template_at<=args.template_age):
                    # Only one usable image is sufficient for a pose estimate,
                    # but it is model-based and NOT a fresh depth measurement.
                    candidates=[]
                    if p0 is not None:
                        w,e=mono_estimate(template,p0,k1,d1,np.eye(3),
                                          np.zeros((3,1)),args.max_mono_reproj)
                        if w is not None: candidates.append((w,e,'MONO_LEFT'))
                    if p1 is not None:
                        w,e=mono_estimate(template,p1,k2,d2,r,t,args.max_mono_reproj)
                        if w is not None: candidates.append((w,e,'MONO_RIGHT'))
                    if candidates:
                        wrist,pnp_err,mode=min(candidates,key=lambda x:x[1])
                        reason='mono_unverified'
                if wrist is not None:
                    # Reset smoothing after invalid gaps; do not invent samples.
                    if smooth is None or now-smooth_at>args.max_gap:
                        smooth=wrist.copy()
                    else:
                        smooth=args.alpha*wrist+(1-args.alpha)*smooth
                    smooth_at=now
                else:
                    smooth=None
                row={'frame':frame_id,'time_s':f'{elapsed:.4f}','mode':mode,
                     'left_detected':int(p0 is not None),'right_detected':int(p1 is not None),
                     'reproj_px':f'{reproj:.3f}' if np.isfinite(reproj) else '',
                     'pnp_px':f'{pnp_err:.3f}' if np.isfinite(pnp_err) else '',
                     'wrist_reproj_px':f'{wrist_err:.3f}' if np.isfinite(wrist_err) else '',
                     'wrist_front':wrist_front,'knuckles_good':knuckles_good,
                     'reason':reason}
                if wrist is not None:
                    for k,v in zip(('x','y','z'),wrist): row[k]=f'{v:.5f}'
                    for k,v in zip(('smooth_x','smooth_y','smooth_z'),smooth): row[k]=f'{v:.5f}'
                writer.writerow(row)
                if frame_id % 30 == 0:
                    f.flush()
                    if wrist is not None:
                        print(f'{frame_id:05d} {mode:10s} wrist={np.round(wrist,2)} err={reproj:.2f}px')
                    else:
                        print(f'{frame_id:05d} INVALID')
                if not args.no_display:
                    text=f'{mode}: '+(' '.join(f'{v:.1f}' for v in smooth) if smooth is not None else 'not tracked')
                    a=draw(a,p0,'LEFT '+text,(0,255,0))
                    b=draw(b,p1,'RIGHT '+mode,(0,255,255))
                    if a.shape[0]!=b.shape[0]:
                        h=min(a.shape[0],b.shape[0]); a=cv2.resize(a,(int(a.shape[1]*h/a.shape[0]),h)); b=cv2.resize(b,(int(b.shape[1]*h/b.shape[0]),h))
                    joined=np.hstack((a,cv2.flip(b,0)))
                    if joined.shape[1]>1800:
                        scale=1800/joined.shape[1]
                        joined=cv2.resize(joined,None,fx=scale,fy=scale)
                    cv2.imshow('Stereo wrist 3D (q to quit)',joined)
                    if cv2.waitKey(1)&0xFF==ord('q'):
                        break
                frame_id+=1
    finally:
        det0.close(); det1.close()
        left.release(); right.release()
        cv2.destroyAllWindows()
        print(f'Saved {frame_id} frames to {args.output}')


if __name__=='__main__':
    main()
