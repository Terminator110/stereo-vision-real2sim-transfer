#!/usr/bin/env python3
"""Audit existing SAM3+FoundationStereo wristband trajectory (no GPU inference).

Panels: 3-D path in table frame, reconstructed table height, and raw Camera1
keyframes with calibrated 3-D wrist XYZ projected back to raw pixels.
Optional --annotate lets you click the SAME physical wristband reference on
five or more keyframes; saves manual-vs-predicted pixel residuals.

A reprojection of the estimator's own XYZ is not independent proof of depth.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path.home() / 'physikal_workflow/dex-5'
TRIAL = ROOT / 'FoundationStereo/wristband_trial_new'
CALIB = ROOT / 'code/stereo/calibration/parallel_results_v3/stereo_calibration.npz'
TABLE = ROOT / 'code/stereo/calibration/parallel_results_v3/table_frame_multi.npz'


def args_parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, default=TRIAL)
    p.add_argument('--calib', type=Path, default=CALIB)
    p.add_argument('--table', type=Path, default=TABLE)
    p.add_argument('--frames', type=int, nargs='+', default=[10,60,100,120,149],
                   help='ZERO-BASED sequence frame numbers, not source video frame numbers')
    p.add_argument('--annotate', action='store_true',help='Click one stable reference point on wristband per keyframe')
    p.add_argument('--out', type=Path, default=None)
    return p.parse_args()


def projected_uv(xyz, K, D):
    xyz=np.asarray(xyz,np.float64).reshape(1,1,3)
    uv,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,D)
    return uv.reshape(2)


def marker_from_fs(u,v,scale, K, D, R1, P1, original_width):
    """Inverse image transform for FS representative pixel (NOT a 3D point)."""
    x_rect=original_width-1-v/scale
    y_rect=u/scale
    direction_rect=np.array([(x_rect-P1[0,2])/P1[0,0],
                             (y_rect-P1[1,2])/P1[1,1],1.0])
    direction_orig=R1.T@direction_rect
    return projected_uv(direction_orig, K, D)


def save_annotations(out, frames, uv_projected, reference):
    fn=out/'manual_wristband_clicks.csv'
    with fn.open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['sequence_frame','clicked_u_px','clicked_v_px','projected_u_px',
                    'projected_v_px','pixel_error_px'])
        for frame, clicked, pred in zip(frames,reference,uv_projected):
            if clicked is None: continue
            w.writerow([frame,*clicked,*pred,np.linalg.norm(np.asarray(clicked)-pred)])
    print('Manual-click comparison:',fn)
    return fn


def main():
    a=args_parse(); trial=a.trial.expanduser().resolve()
    out=(a.out or trial/'wrist_reprojection_diagnostics').expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((trial/'meta.json').read_text())
    take=Path(meta['take']).expanduser()
    if not take.is_absolute(): take=(trial/take).resolve()
    video=take/'camera1.avi'
    with (trial/'wrist_xyz_camera1.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows: raise ValueError('Wrist CSV is empty')
    with np.load(a.calib) as z:
        K=np.asarray(z['K1'],np.float64); D=np.asarray(z['D1'],np.float64)
        R1=np.asarray(z['R1'],np.float64); P1=np.asarray(z['P1'],np.float64)
        w,h=map(int,z['image_size'])
    with np.load(a.table) as z:
        Rt=np.asarray(z['R_cam1_to_table'],np.float64)
        tt=np.asarray(z['t_cam1_to_table'],np.float64).reshape(3)
    if Path(meta['calib']).expanduser().resolve()!=a.calib.expanduser().resolve():
        raise ValueError('Trial metadata references a different stereo calibration!')
    requested=list(dict.fromkeys(a.frames))
    selected=[f for f in requested if 0<=f<len(rows)]
    if not selected: raise ValueError('No keyframes are within wrist trajectory length')
    if len(selected)!=len(requested):print('Warning: some requested frames are out of range')
    data=[]; uv=[]; xyz=[]; txyz=[]; image_frames=[]; image_source=[]; fs_uv=[]
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise FileNotFoundError(f'Could not open {video}')
    for i,row in enumerate(rows):
        pt=np.array([float(row[k]) for k in ('X_cam1_m','Y_cam1_m','Z_cam1_m')])
        xyz.append(pt)
        txyz.append(Rt@pt+tt if np.isfinite(pt).all() else np.full(3,np.nan))
        uv.append(projected_uv(pt,K,D) if np.isfinite(pt).all() and pt[2]>0 else np.full(2,np.nan))
    xyz=np.asarray(xyz); txyz=np.asarray(txyz);uv=np.asarray(uv)
    scale=float(meta['scale'])
    for frame in selected:
        row=rows[frame]
        source=int(row['source_video_frame'])
        cap.set(cv2.CAP_PROP_POS_FRAMES,source)
        ok,bgr=cap.read()
        if not ok:raise IOError(f'Cannot read original frame {source} from {video}')
        if bgr.shape[:2]!=(h,w):raise ValueError(f'Frame {source} shape disagrees with calibration')
        image_frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        image_source.append(source)
        u_fs=float(row['u_fs_px']);v_fs=float(row['v_fs_px'])
        fs_uv.append(marker_from_fs(u_fs,v_fs,scale,K,D,R1,P1,w)
                     if np.isfinite([u_fs,v_fs]).all() else np.full(2,np.nan))
    cap.release()

    # Optional manual annotation against the same physical point in each frame.
    clicks=[None]*len(selected)
    if a.annotate:
        print('\nClick the SAME physical spot (e.g. wristband edge centre) on every image.')
        print('Click once in each image window; close or press Enter to skip. Cropping/zoom is allowed.')
        for k,(frame,img) in enumerate(zip(selected,image_frames)):
            fig,ax=plt.subplots(figsize=(11,6))
            ax.imshow(img);ax.set_title(f'Sequence {frame} / video frame {image_source[k]}: click wristband reference; Enter skips')
            ax.set_xlim(0,w);ax.set_ylim(h,0);ax.grid(alpha=.15)
            pts=plt.ginput(1,timeout=0)
            if pts:clicks[k]=np.asarray(pts[0])
            plt.close(fig)
        save_annotations(out,selected,uv[selected],clicks)

    # One fixed plot showing the entire trajectory as xy at table coordinates, NOT robot world.
    f=plt.figure(figsize=(15,10))
    grid=f.add_gridspec(2,2,height_ratios=[1.2,1.0])
    ax=f.add_subplot(grid[0,0]);az=f.add_subplot(grid[0,1]);ay=f.add_subplot(grid[1,:])
    sc=ax.scatter(1000*txyz[:,0],1000*txyz[:,1],c=np.arange(len(rows)),cmap='viridis',s=13)
    ax.plot(1000*txyz[:,0],1000*txyz[:,1],lw=.6,color='gray',alpha=.5)
    for i in selected:
        if np.isfinite(txyz[i]).all(): ax.annotate(str(i),(txyz[i,0]*1000,txyz[i,1]*1000),fontsize=9)
    ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='Table X (mm)',ylabel='Table Y (mm)',title='Reconstructed wristband path: table XY')
    f.colorbar(sc,ax=ax,label='Sequence frame')
    az.plot(np.arange(len(rows)),1000*txyz[:,2],label='Wristband estimated table-Z',lw=1.5)
    for i in selected:az.axvline(i,linestyle=':',alpha=.4)
    az.set(xlabel='Sequence frame',ylabel='Height / table Z (mm)',title='Wristband estimated Z in calibrated table frame')
    az.grid(alpha=.25)
    for c,label in enumerate(['X','Y','Z']):ay.plot(np.arange(len(rows)),1000*txyz[:,c],label=f'Table {label}')
    for i in selected:ay.axvline(i,linestyle=':',alpha=.3)
    ay.set(xlabel='Sequence frame',ylabel='Position (mm)',title='Time-series in table coordinates (NOT MuJoCo offsets)')
    ay.legend(ncol=3);ay.grid(alpha=.25)
    f.tight_layout();f.savefig(out/'wrist_3d_path_and_height.png',dpi=160);plt.close(f)

    # Raw video snapshots. Red = 3D reprojected; cyan = depth-mask representative 2D point.
    cols=min(2,len(selected));nrows=int(np.ceil(len(selected)/cols))
    fig,axes=plt.subplots(nrows,cols,figsize=(9*cols,5*nrows),squeeze=False)
    for k,frame in enumerate(selected):
        ax=axes.flat[k];img=image_frames[k]; p=uv[frame]; q=fs_uv[k]
        ax.imshow(img)
        if np.isfinite(p).all():
            ax.scatter([p[0]],[p[1]],marker='+',c=['red'],s=260,lw=2.8,label='3D XYZ reprojected')
            ax.add_patch(plt.Circle(tuple(p),12,color='red',fill=False,lw=1.3))
        if np.isfinite(q).all():ax.scatter([q[0]],[q[1]],marker='x',c=['cyan'],s=110,lw=2.5,label='Original depth/mask estimator 2D point')
        if clicks[k] is not None:
            ax.scatter([clicks[k][0]],[clicks[k][1]],c=['yellow'],marker='o',s=85,edgecolors='black',label='Manually clicked reference')
            ax.plot([p[0],clicks[k][0]],[p[1],clicks[k][1]],'y--',lw=1.3)
        ax.set(xlim=(0,w),ylim=(h,0),title=f'Seq {frame} | Source frame {image_source[k]} | Z(table)={txyz[frame,2]*1000:.1f} mm')
        ax.legend(loc='upper right',fontsize=8);ax.set_xlabel('Original Camera1 pixel X');ax.set_ylabel('Original Camera1 pixel Y')
    for ax in axes.flat[len(selected):]:ax.axis('off')
    fig.suptitle('Wristband check: reprojected 3D estimator versus source video (not independent depth proof)',fontsize=14)
    fig.tight_layout();fig.savefig(out/'wrist_keyframes_reprojected.png',dpi=155);plt.close(fig)

    # Quantify mask estimator pixel location vs reprojected median 3D.
    print('\nSeq  Source   Uproject  Vproject   U_FSmedian  V_FSmedian  Delta2D(px)  Z_table(mm)  ClickError(px)')
    for k,frame in enumerate(selected):
        p,q=uv[frame],fs_uv[k]
        delta=np.linalg.norm(p-q) if np.isfinite([*p,*q]).all() else np.nan
        err=np.linalg.norm(p-clicks[k]) if clicks[k] is not None else np.nan
        print(f'{frame:3d} {image_source[k]:7d} {p[0]:9.1f} {p[1]:9.1f} {q[0]:11.1f} {q[1]:11.1f} {delta:12.1f} {txyz[frame,2]*1000:12.1f} {err:14.1f}')
    print('\nSaved:',out/'wrist_keyframes_reprojected.png')
    print('Saved:',out/'wrist_3d_path_and_height.png')
    print('CAUTION: 3D XYZ and FS-median originate from the same stereo/mask estimate.')
    print('Pixel agreement is a correspondence check, NOT independent validation of depth or table height.')
    print('The physical tabletop has Z=0 only if table calibration defines its origin at the surface.')

if __name__=='__main__':main()
