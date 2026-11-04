#!/usr/bin/env python3
"""Inspect triangulated AprilTag 4/5 centres from stereo_wrist_parallel.py.
Coordinates are camera-1 frame, metres. Each tag has its own trajectory.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, default=Path('stereo/parallel_wrist_tracking.csv'))
    ap.add_argument('--out', type=Path, default=Path('stereo/parallel_wrist_visualization'))
    ap.add_argument('--show', action='store_true', help='Open interactive 3D plot locally')
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    needed = {'frame','time_s','tag_id','valid','x_cam1_m','y_cam1_m','z_cam1_m'}
    if not needed.issubset(df.columns):
        raise SystemExit('Missing columns: ' + ', '.join(sorted(needed - set(df.columns))))
    args.out.mkdir(parents=True, exist_ok=True)
    for c in ['frame','time_s','tag_id','valid','x_cam1_m','y_cam1_m','z_cam1_m']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    valid = df[(df.valid == 1) & df[['x_cam1_m','y_cam1_m','z_cam1_m']].notna().all(axis=1)].copy()
    valid = valid.sort_values(['tag_id','time_s'])
    total_frames = df.frame.nunique()
    frames_with_pos = valid.frame.nunique()
    print(f'Frames with at least one reconstructed tag: {frames_with_pos}/{total_frames} '
          f'({100*frames_with_pos/max(total_frames,1):.1f}%)')
    print('Total valid tag observations:', len(valid))
    for tag in sorted(df.tag_id.dropna().unique()):
        v = valid[valid.tag_id == tag]
        print(f'Tag {int(tag)}: valid observations={len(v)}, '
              f'XYZ range(mm)=' + str(np.round((v[['x_cam1_m','y_cam1_m','z_cam1_m']].max() - v[['x_cam1_m','y_cam1_m','z_cam1_m']].min()).to_numpy()*1000,1).tolist()) if len(v) else f'Tag {int(tag)}: no valid positions')
    if valid.empty:
        raise SystemExit('No valid triangulated 3D positions to visualize.')

    palette = {4:'tab:orange',5:'tab:blue'}
    coords = ['x_cam1_m','y_cam1_m','z_cam1_m']
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111,projection='3d')
    for tag in sorted(valid.tag_id.unique()):
        sub = valid[valid.tag_id == tag]
        v = sub[coords].to_numpy()*1000
        color = palette.get(int(tag),'tab:green')
        ax.scatter(v[:,0],v[:,1],v[:,2],s=10,alpha=.65,color=color,label=f'AprilTag {int(tag)} ({len(sub)} points)')
        # Draw contiguous runs only; don't draw lines across missing detections.
        groups = (sub.frame.diff().fillna(1)!=1).cumsum()
        for _, run in sub.groupby(groups):
            if len(run)>1:
                r = run[coords].to_numpy()*1000
                ax.plot(r[:,0],r[:,1],r[:,2],color=color,linewidth=1,alpha=.65)
    all_xyz=valid[coords].to_numpy()*1000
    xyz_min,xyz_max=all_xyz.min(axis=0),all_xyz.max(axis=0)
    center=(xyz_min+xyz_max)/2
    span=max(float(np.max(xyz_max-xyz_min)),50.0)
    for setter, c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):
        setter(c-span*.55,c+span*.55)
    ax.set_xlabel('Camera 1 X (mm)')
    ax.set_ylabel('Camera 1 Y (mm)')
    ax.set_zlabel('Camera 1 Z / depth (mm)')
    ax.set_title('Stereo AprilTag centres — Camera 1 frame (not wrist joint)')
    ax.legend()
    fig.tight_layout()
    path3=args.out/'trajectory_3d.png'
    fig.savefig(path3,dpi=160)

    fig2,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for tag in sorted(valid.tag_id.unique()):
        sub=valid[valid.tag_id==tag]
        t=sub.time_s.to_numpy()
        color=palette.get(int(tag),'tab:green')
        for j,coord in enumerate(coords):
            y=sub[coord].to_numpy()*1000
            axs[j].scatter(t,y,s=9,alpha=.75,color=color,label=f'Tag {int(tag)}' if j==0 else None)
            # Avoid connecting gaps; connect only contiguous frame runs
            groups=(sub.frame.diff().fillna(1)!=1).cumsum()
            for _,run in sub.groupby(groups):
                if len(run)>1:
                    axs[j].plot(run.time_s,run[coord]*1000,color=color,linewidth=1,alpha=.6)
    for ax,label in zip(axs,('X (mm)','Y (mm)','Z / depth (mm)')):
        ax.set_ylabel(label)
        ax.grid(alpha=.3)
    axs[0].legend(loc='best')
    axs[-1].set_xlabel('Time (s)')
    fig2.suptitle('Stereo marker positions vs time — gaps indicate no valid stereo observation')
    fig2.tight_layout()
    path_t=args.out/'xyz_vs_time.png'
    fig2.savefig(path_t,dpi=160)
    print('Saved:',path3)
    print('Saved:',path_t)
    if args.show:
        plt.show()
    else:
        plt.close('all')

if __name__=='__main__':
    main()
