#!/usr/bin/env python3
"""Offline stereo wrist AprilTag trajectories and object keyframe extraction.

Run with --take stereo/recordings/take_... . Outputs in take/processed.
Uses RAW, unflipped AVI frames and timestamps.csv from stereo_dual_recorder.py.
Tag centre is NOT anatomical wrist. Calibration table axes are arbitrary.
"""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

def load_timing(path):
    rows = []
    if path.exists():
        with path.open(newline='') as f:
            for r in csv.DictReader(f):
                try:
                    rows.append((int(r['frame']), (float(r['cam1_retrieved_elapsed_s']) + float(r['cam2_retrieved_elapsed_s'])) / 2.0,
                                 abs(float(r['cam1_retrieved_elapsed_s']) - float(r['cam2_retrieved_elapsed_s'])) * 1000.0))
                except (KeyError, ValueError):
                    pass
    return {idx:(ts, offset) for idx,ts,offset in rows}

def project_err(x, uv, K, D, R, t):
    rv,_ = cv2.Rodrigues(R)
    p,_ = cv2.projectPoints(x.astype(np.float64), rv, t, K, D)
    return np.linalg.norm(p.reshape(-1,2) - uv, axis=1)

def triangulate(a, b, c):
    u1 = cv2.undistortPoints(a.reshape(-1,1,2),c['K1'],c['D1']).reshape(-1,2)
    u2 = cv2.undistortPoints(b.reshape(-1,1,2),c['K2'],c['D2']).reshape(-1,2)
    p1=np.hstack((np.eye(3),np.zeros((3,1))))
    p2=np.hstack((c['R'],c['T'].reshape(3,1)))
    h=cv2.triangulatePoints(p1,p2,u1.T,u2.T)
    if not np.all(np.abs(h[3])>1e-12):
        return None
    xyz=(h[:3]/h[3]).T
    if not np.isfinite(xyz).all() or np.any(xyz[:,2]<=0) or np.any((c['R']@xyz.T+c['T'].reshape(3,1))[2]<=0):
        return None
    e1=project_err(xyz,a,c['K1'],c['D1'],np.eye(3),np.zeros((3,1)))
    e2=project_err(xyz,b,c['K2'],c['D2'],c['R'],c['T'].reshape(3,1))
    edge=np.linalg.norm(xyz-np.roll(xyz,-1,axis=0),axis=1)*1000
    return xyz, float(max(np.max(e1),np.max(e2))),float(np.mean(edge)),float(np.ptp(edge))

def detect(img,det,ids):
    cs,ts,_=det.detectMarkers(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    if ts is None: return {}
    return {int(t):np.asarray(c,np.float64).reshape(4,2) for c,t in zip(cs,ts.ravel()) if int(t) in ids}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--take',type=Path,required=True,help='Folder containing camera1.avi, camera2.avi, timestamps.csv')
    ap.add_argument('--calib',type=Path,default=Path('stereo/calibration/results_new/stereo_calibration.npz'))
    ap.add_argument('--table',type=Path,default=Path('stereo/calibration/results_new/table_frame_multi.npz'))
    ap.add_argument('--ids',type=int,nargs='+',default=[4,5])
    ap.add_argument('--tag-mm',type=float,default=25.0)
    ap.add_argument('--max-reproj',type=float,default=3.0)
    ap.add_argument('--max-edge-error',type=float,default=0.20)
    ap.add_argument('--object-every-s',type=float,default=0.0,help='Extract synchronized image pairs for later SAM processing; 0 disables')
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args()
    if args.tag_mm<=0: ap.error('--tag-mm must be positive')
    out=args.out or args.take/'processed'
    out.mkdir(parents=True,exist_ok=True)
    with np.load(args.calib) as z:
        c={k:np.asarray(z[k],np.float64) for k in ['K1','D1','K2','D2','R','T']}
    with np.load(args.table) as z:
        Rt=np.asarray(z['R_cam1_to_table'],np.float64)
        tt=np.asarray(z['t_cam1_to_table'],np.float64).reshape(3)
    ca=cv2.VideoCapture(str(args.take/'camera1.avi'))
    cb=cv2.VideoCapture(str(args.take/'camera2.avi'))
    if not ca.isOpened() or not cb.isOpened():
        raise SystemExit('Cannot open camera1.avi and camera2.avi in --take')
    fps=ca.get(cv2.CAP_PROP_FPS) or 20.0
    fps2=cb.get(cv2.CAP_PROP_FPS) or fps
    n1=int(ca.get(cv2.CAP_PROP_FRAME_COUNT)); n2=int(cb.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video frames: cam1={n1}, cam2={n2}; encoded fps={fps:.3f}/{fps2:.3f}')
    if n1!=n2: print('WARNING: frame counts differ; processing shorter stream')
    timing=load_timing(args.take/'timestamps.csv')
    if not timing: print('WARNING: missing timestamps.csv, using encoded frame index/fps (not real time)')
    det=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    fields=['frame','time_s','camera_pair_offset_ms','selected_tag_id','valid','reason',
            'tag_x_cam1_m','tag_y_cam1_m','tag_z_cam1_m','tag_x_table_m','tag_y_table_m','tag_z_table_m',
            'reproj_max_px','edge_mean_mm','edge_range_mm','shared_ids','valid_ids']
    for tag in args.ids:
        fields += [f'id{tag}_valid',f'id{tag}_x_table_m',f'id{tag}_y_table_m',f'id{tag}_z_table_m',f'id{tag}_reproj_px']
    n=good=0; last_object_time=-1e12; reasons={}; choices={}
    frames_dir=out/'object_keyframes'
    with (out/'wrist_tags_table.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader()
        while True:
            oka,a=ca.read(); okb,b=cb.read()
            if not oka or not okb: break
            time_s,offset_ms=timing.get(n,(n/fps,float('nan')))
            if args.object_every_s>0 and time_s-last_object_time>=args.object_every_s-1e-6:
                frames_dir.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(frames_dir/f'frame_{n:06d}_camera1.png'),a)
                cv2.imwrite(str(frames_dir/f'frame_{n:06d}_camera2.png'),b)
                last_object_time=time_s
            left=detect(a,det,args.ids);right=detect(b,det,args.ids)
            shared=sorted(set(left)&set(right))
            valids={}
            for tag in shared:
                res=triangulate(left[tag],right[tag],c)
                if res is None: continue
                pts,reproj,edge,spread=res
                if reproj>args.max_reproj or abs(edge-args.tag_mm)>args.tag_mm*args.max_edge_error or spread>8.0: continue
                pos=pts.mean(axis=0); table=Rt@pos+tt
                valids[tag]=(pos,table,reproj,edge,spread)
            row=dict.fromkeys(fields,'')
            row.update(frame=n,time_s=f'{time_s:.6f}',camera_pair_offset_ms=f'{offset_ms:.3f}',
                       valid=0,reason='no_shared_id' if not shared else 'geometry_rejected',
                       shared_ids=';'.join(map(str,shared)),valid_ids=';'.join(map(str,sorted(valids))))
            for tag,(_,p,e,_,_) in valids.items():
                row.update({f'id{tag}_valid':1,f'id{tag}_x_table_m':f'{p[0]:.6f}',
                            f'id{tag}_y_table_m':f'{p[1]:.6f}',f'id{tag}_z_table_m':f'{p[2]:.6f}',
                            f'id{tag}_reproj_px':f'{e:.3f}'})
            for tag in args.ids:
                if tag not in valids: row[f'id{tag}_valid']=0
            if valids:
                tag=min(valids,key=lambda k:valids[k][2]);p,pt,e,edge,spread=valids[tag]
                row.update(selected_tag_id=tag,valid=1,reason='ok',
                           tag_x_cam1_m=f'{p[0]:.6f}',tag_y_cam1_m=f'{p[1]:.6f}',tag_z_cam1_m=f'{p[2]:.6f}',
                           tag_x_table_m=f'{pt[0]:.6f}',tag_y_table_m=f'{pt[1]:.6f}',tag_z_table_m=f'{pt[2]:.6f}',
                           reproj_max_px=f'{e:.3f}',edge_mean_mm=f'{edge:.2f}',edge_range_mm=f'{spread:.2f}')
                good+=1;choices[tag]=choices.get(tag,0)+1
            else: reasons[row['reason']]=reasons.get(row['reason'],0)+1
            writer.writerow(row);n+=1
            if n%200==0:print(f'frames={n} valid={good} ({good/n:.1%})')
    ca.release();cb.release()
    print('Saved:',out/'wrist_tags_table.csv')
    print(f'Valid {good}/{n} ({good/max(n,1):.1%}); selected tag counts={choices}; rejection={reasons}')
    if args.object_every_s>0:print('Extracted object keyframe pairs:',frames_dir)
    print('WARNING: selected_tag XYZ is TAG CENTER, not common wrist point. Tag switches cause jumps.')
    print('WARNING: no cup/coaster dynamic pose or wrist orientation is inferred by this script.')

if __name__=='__main__':main()
