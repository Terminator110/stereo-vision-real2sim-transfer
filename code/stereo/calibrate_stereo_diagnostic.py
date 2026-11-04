#!/usr/bin/env python3
"""Diagnostic ChArUco stereo calibration; never overwrites previous calibration results.
Use only original, unflipped left_###/right_### pairs.
"""
import argparse
import csv
from pathlib import Path
import re
import cv2
import numpy as np

SX,SY=7,10
SQUARE=0.02425
MARKER=0.018
DICT=cv2.aruco.DICT_6X6_250

def discover(folder):
    items={}
    for p in folder.iterdir():
        m=re.fullmatch(r'(left|right)_(\d+)\.(?:png|jpe?g)',p.name,re.I)
        if m: items.setdefault(m.group(2),{})[m.group(1).lower()]=p
    return [(k, v['left'],v['right']) for k,v in sorted(items.items(),key=lambda x:int(x[0])) if {'left','right'}<=set(v)]

def detect(detector, image, board_pts):
    cc,ii,_,_=detector.detectBoard(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    if cc is None or ii is None: return {}
    cc=np.asarray(cc,dtype=np.float32).reshape(-1,2)
    ii=np.asarray(ii,dtype=int).reshape(-1)
    if len(cc)!=len(ii) or len(set(ii))!=len(ii) or any(x<0 or x>=len(board_pts) for x in ii): return {}
    return dict(zip(map(int,ii),cc))

def fit_camera(rows, idx, size, flags):
    oo=[]; ii=[]
    for row in rows:
        d=row['det'][idx]
        ids=sorted(d)
        oo.append(row['board'][ids].reshape(-1,1,3))
        ii.append(np.asarray([d[k] for k in ids],np.float32).reshape(-1,1,2))
    K0=np.array([[size[0],0,(size[0]-1)/2],[0,size[0],(size[1]-1)/2],[0,0,1]],dtype=np.float64)
    D0=np.zeros((5,1),np.float64)
    rms,K,D,rv,tv=cv2.calibrateCamera(oo,ii,size,K0,D0,flags=flags,criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,200,1e-9))
    errs={}
    for row,ob,im,r,t in zip(rows,oo,ii,rv,tv):
        projected,_=cv2.projectPoints(ob,r,t,K,D)
        errs[row['label']]=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-im.reshape(-1,2))**2,axis=1))))
    return rms,K,D,errs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pairs',type=Path,default=Path(__file__).resolve().parent/'calibration'/'pairs')
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'calibration'/'diagnostic_results')
    ap.add_argument('--legacy',action='store_true')
    ap.add_argument('--min-shared',type=int,default=15)
    ap.add_argument('--max-view-error',type=float,default=2.0,help='Px threshold for pruning; diagnostic heuristic, not accuracy certification')
    ap.add_argument('--max-remove',type=int,default=10)
    args=ap.parse_args()
    board=cv2.aruco.CharucoBoard((SX,SY),SQUARE,MARKER,cv2.aruco.getPredefinedDictionary(DICT))
    if args.legacy: board.setLegacyPattern(True)
    det=cv2.aruco.CharucoDetector(board)
    bp=np.asarray(board.getChessboardCorners(),np.float32)
    files=discover(args.pairs)
    if not files: raise SystemExit('No left_### / right_### images in '+str(args.pairs))
    rows=[]; size=None
    for label,lf,rf in files:
        a,b=cv2.imread(str(lf)),cv2.imread(str(rf))
        if a is None or b is None or a.shape[:2]!=b.shape[:2]: print('SKIP',label,'invalid pair'); continue
        s=(a.shape[1],a.shape[0]); size=size or s
        if s!=size: print('SKIP',label,'size mismatch'); continue
        da,db=detect(det,a,bp),detect(det,b,bp)
        common=sorted(set(da)&set(db))
        print(f'PAIR {label}: left={len(da)} right={len(db)} shared={len(common)}')
        if len(common)<args.min_shared or min(len(da),len(db))<12: continue
        rows.append(dict(label=label, det=(da,db),common=common,board=bp))
    if len(rows)<12: raise SystemExit('Fewer than 12 usable pairs. Check board/capture.')
    # Square pixels, centered principal point, and a simpler radial model reduce unstable freedoms.
    flags=(cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_ASPECT_RATIO|
           cv2.CALIB_FIX_PRINCIPAL_POINT|cv2.CALIB_ZERO_TANGENT_DIST|
           cv2.CALIB_FIX_K3|cv2.CALIB_FIX_K4|cv2.CALIB_FIX_K5|cv2.CALIB_FIX_K6)
    active=rows[:]; removed=[]
    while True:
        fits=[fit_camera(active,i,size,flags) for i in (0,1)]
        ranked=sorted(((max(fits[0][3][r['label']],fits[1][3][r['label']]),r['label']) for r in active),reverse=True)
        worst,wl=ranked[0]
        if worst<=args.max_view_error or len(active)<=15 or len(removed)>=args.max_remove: break
        active=[r for r in active if r['label']!=wl]; removed.append((wl,worst))
        print(f'PRUNE pair {wl}: worst monocular RMS {worst:.3f}px')
    (r1,K1,D1,e1),(r2,K2,D2,e2)=fits
    print('\n===== PER-PAIR REPROJECTION ERROR (px) =====')
    print('Pair  C1     C2')
    for row in active:
        label=row['label']; print(f'{label:>4}  {e1[label]:6.3f} {e2[label]:6.3f}')
    for i,(r,K,D,_) in enumerate(fits,1):
        print(f'\nCAMERA {i} RMS: {r:.3f}px\nK=\n{K}\nD={D.ravel()}')
    print('Excluded:',removed)
    oo=[]; ii1=[]; ii2=[]
    for row in active:
        ids=row['common']; oo.append(bp[ids].reshape(-1,1,3).copy())
        ii1.append(np.asarray([row['det'][0][k] for k in ids],np.float32).reshape(-1,1,2))
        ii2.append(np.asarray([row['det'][1][k] for k in ids],np.float32).reshape(-1,1,2))
    sr,_,_,_,_,R,T,E,F=cv2.stereoCalibrate(oo,ii1,ii2,K1,D1,K2,D2,size,flags=cv2.CALIB_FIX_INTRINSIC,criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,150,1e-8))
    print(f'\nSTEREO RMS: {sr:.3f}px; estimated baseline: {np.linalg.norm(T)*1000:.1f}mm')
    print('Used',len(active),'pairs; omitted',len(files)-len(active),'pairs overall')
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'per_pair_errors.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['pair','cam1_error_px','cam2_error_px','shared_corners'])
        for row in active: w.writerow([row['label'],e1[row['label']],e2[row['label']],len(row['common'])])
    # Diagnostic only: avoid confusing these with prior validated production calibration.
    np.savez(args.out/'STEREO_DIAGNOSTIC_NOT_VALIDATED.npz', K1=K1,D1=D1,K2=K2,D2=D2,R=R,T=T,stereo_rms=sr,image_size=size,used_pairs=np.array([r['label'] for r in active]))
    print('Diagnostics:',args.out)
    print('NOT VALIDATED: Compare physical baseline; inspect rectification & known-distance errors before using for tracking.')

if __name__=='__main__': main()
