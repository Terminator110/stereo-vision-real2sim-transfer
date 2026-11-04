#!/usr/bin/env python3
"""Read-only comparison of original and swapped DEX5 FoundationStereo/MuJoCo runs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STAGES = [
    ('camera_xyz', 'wrist_xyz_camera1.csv', ['X_cam1_m','Y_cam1_m','Z_cam1_m']),
    ('rectified_xyz', 'wrist_xyz_camera1.csv', ['X_rect_m','Y_rect_m','Z_rect_m']),
    ('table_xyz', 'palm_pose_6dof.csv', ['X_table_m','Y_table_m','Z_table_m']),
    ('mujoco_target', 'sim_quality.csv', ['wrist_target_x','wrist_target_y','wrist_target_z']),
    ('mujoco_actual', 'sim_quality.csv', ['wrist_actual_x','wrist_actual_y','wrist_actual_z']),
]

def xyz(df, cols):
    if not all(k in df for k in cols):
        return None
    a = df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float, copy=True)
    a[~np.isfinite(a).all(axis=1)] = np.nan
    return a

def stats(name, arr, mask=None):
    if arr is None: return f'{name}: columns unavailable'
    valid = np.isfinite(arr).all(axis=1)
    if mask is not None and len(mask)==len(valid): valid &= mask
    a = arr[valid]
    if not len(a): return f'{name}: NO valid points'
    diff = np.linalg.norm(np.diff(a, axis=0), axis=1) if len(a)>1 else np.array([])
    bounds = a.max(axis=0)-a.min(axis=0)
    result = (f'{name}: valid={len(a)}/{len(arr)}; min={np.round(a.min(axis=0),4)} m; '
              f'max={np.round(a.max(axis=0),4)} m; span={np.round(bounds*1000,1)} mm; '
              f'median={np.round(np.median(a,axis=0),4)} m')
    if diff.size: result += f'; consecutive-valid-step median={np.median(diff)*1000:.1f} mm, p95={np.percentile(diff,95)*1000:.1f} mm, max={diff.max()*1000:.1f} mm (gaps may contribute)'
    return result

def load_csv(path):
    if not path.is_file(): return None
    return pd.read_csv(path)

def set_equal_3d(ax, arrays):
    v=np.concatenate([a[np.isfinite(a).all(axis=1)] for a in arrays if a is not None and np.isfinite(a).all(axis=1).any()],axis=0) if any(a is not None and np.isfinite(a).all(axis=1).any() for a in arrays) else None
    if v is None:return
    lo=v.min(axis=0); hi=v.max(axis=0); center=(lo+hi)/2; radius=max(np.max(hi-lo)/2,0.05)*1.1
    ax.set_xlim(center[0]-radius,center[0]+radius)
    ax.set_ylim(center[1]-radius,center[1]+radius)
    ax.set_zlim(center[2]-radius,center[2]+radius)
    ax.set_box_aspect([1,1,1]); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

def plot_stage(out, stage, original, swapped):
    if original is None and swapped is None:return
    fig=plt.figure(figsize=(14,6)); ax=fig.add_subplot(121,projection='3d'); ax2=fig.add_subplot(122)
    for name,arr in [('Original',original),('Swapped',swapped)]:
        if arr is None:continue
        ok=np.isfinite(arr).all(axis=1)
        if not ok.any():continue
        # NaNs make line breaks where reconstruction is invalid
        ax.plot(arr[:,0],arr[:,1],arr[:,2],label=name,linewidth=1.2)
        first=arr[np.flatnonzero(ok)[0]]; last=arr[np.flatnonzero(ok)[-1]]
        ax.scatter(*first,marker='o',s=40); ax.scatter(*last,marker='x',s=55)
        ax2.plot(arr[:,0],arr[:,1],label=name,linewidth=1.4)
        ax2.scatter([first[0]],[first[1]],marker='o',s=35)
    set_equal_3d(ax,[original,swapped]); ax.legend(); ax.set_title(stage+' | 3D (start circle, end X)')
    ax2.set_xlabel('X (m)');ax2.set_ylabel('Y (m)');ax2.set_aspect('equal',adjustable='datalim');ax2.grid(alpha=.3);ax2.legend();ax2.set_title(stage+' | XY top view')
    fig.tight_layout();fig.savefig(out/(stage+'_3d_xy.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(14,10),sharex=False)
    for name,arr in [('Original',original),('Swapped',swapped)]:
        if arr is None:continue
        for i in range(3): axes[i].plot(np.arange(len(arr)),arr[:,i],label=name,linewidth=1)
        d=np.full(len(arr),np.nan)
        if len(arr)>1:
            good=np.isfinite(arr[1:]).all(axis=1)&np.isfinite(arr[:-1]).all(axis=1)
            d[1:][good]=np.linalg.norm(np.diff(arr,axis=0)[good],axis=1)*1000
        axes[3].plot(np.arange(len(arr)),d,label=name,linewidth=1)
    for i,label in enumerate(['X (m)','Y (m)','Z (m)','Step (mm/frame)']):axes[i].set_ylabel(label);axes[i].grid(alpha=.3);axes[i].legend()
    axes[3].set_xlabel('Sequence row (check source-frame alignment)')
    fig.suptitle(stage+' | Axis motion and valid adjacent-frame jumps')
    fig.tight_layout();fig.savefig(out/(stage+'_timeseries.png'),dpi=160);plt.close(fig)

def plot_tracking(out, data):
    fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True)
    have=False
    for run,df in data.items():
        if df is None:continue
        for field,ax in [('position_error_mm',axes[0]),('orientation_error_deg',axes[1])]:
            if field in df:
                ax.plot(pd.to_numeric(df[field],errors='coerce').to_numpy(),label=run+' '+field,linewidth=1); have=True
    if have:
        axes[0].set_ylabel('Position error (mm)');axes[1].set_ylabel('Rotation error (deg)');axes[1].set_xlabel('Sequence row')
        for ax in axes:ax.grid(alpha=.3);ax.legend()
        fig.tight_layout();fig.savefig(out/'ik_errors.png',dpi=160)
    plt.close(fig)

def plot_orientation(out, qs):
    from scipy.spatial.transform import Rotation
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True)
    found=False
    for run,arr in qs.items():
        if arr is None:continue
        valid=np.isfinite(arr).all(axis=1)&(np.linalg.norm(np.nan_to_num(arr),axis=1)>0.01)
        if not valid.any():continue
        euler=np.full((len(arr),3),np.nan)
        euler[valid]=Rotation.from_quat(arr[valid]).as_euler('xyz',degrees=True)
        for i in range(3):axes[i].plot(euler[:,i],label=run,linewidth=1)
        found=True
    if found:
        for ax,label in zip(axes,['Euler X (deg)','Euler Y (deg)','Euler Z (deg)']):ax.set_ylabel(label);ax.legend();ax.grid(alpha=.3)
        axes[-1].set_xlabel('Sequence row');fig.suptitle('MuJoCo target quaternion, XYZW; Euler XYZ display only (wrap/gimbal effects possible)')
        fig.tight_layout();fig.savefig(out/'target_orientation_euler.png',dpi=160)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--take',default='take_20261104_163208_670689')
    p.add_argument('--project',type=Path,default=Path.home()/'physikal_workflow/dex-5')
    p.add_argument('--out',type=Path,default=Path('/tmp/dex5_reconstruction_compare'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    take=a.take.removesuffix('_swapped')
    runs={'Original':take,'Swapped':take+'_swapped'}
    dfs={};qs={};report=[]
    for run,t in runs.items():
        trial=a.project/'FoundationStereo'/('wristband_batch_'+t)
        sim=a.project/'outputs'/('smoothed_'+t)
        dfs[run]={
            'wrist_xyz_camera1.csv':load_csv(trial/'wrist_xyz_camera1.csv'),
            'palm_pose_6dof.csv':load_csv(trial/'palm_pose_6dof.csv'),
            'sim_quality.csv':load_csv(sim/'sim_quality.csv'),
        }
        for fname,df in dfs[run].items():
            if df is None:report.append(f'MISSING {run}: {fname}')
            else:report.append(f'{run}: {fname} -> {len(df)} rows; columns={list(df.columns)}')
        npz=sim/'sim_qpos.npz'
        if npz.is_file():
            with np.load(npz,allow_pickle=False) as z:
                qs[run]=np.array(z['target_quat_xyzw'],dtype=float) if 'target_quat_xyzw' in z else None
        else:qs[run]=None
        w=dfs[run]['wrist_xyz_camera1.csv']
        if w is not None and 'status' in w:report.append(f'{run} depth status: {w.status.value_counts(dropna=False).to_dict()}')
        s=dfs[run]['sim_quality.csv']
        if s is not None and 'ik_status' in s:report.append(f'{run} IK status: {s.ik_status.value_counts(dropna=False).to_dict()}')
        if s is not None:
            for f in ['position_error_mm','orientation_error_deg','filtered_target_delta_mm']:
                if f in s:
                    v=pd.to_numeric(s[f],errors='coerce').to_numpy(dtype=float)
                    v=v[np.isfinite(v)]
                    if len(v): report.append(f'{run} {f}: median={np.median(v):.2f}, p95={np.percentile(v,95):.2f}, max={v.max():.2f}')
    for stage,filename,cols in STAGES:
        arrays={run:xyz(dfs[run][filename],cols) if dfs[run][filename] is not None else None for run in runs}
        for run,arr in arrays.items():
            mask=None
            df=dfs[run][filename]
            if df is not None and 'status' in df and stage in ('camera_xyz','rectified_xyz'):
                mask=(df['status'].astype(str).to_numpy()=='VALID')
                if arr is not None:arr=arr.copy();arr[~mask]=np.nan;arrays[run]=arr
            report.append(stats(run+' '+stage,arr))
        plot_stage(a.out,stage,arrays['Original'],arrays['Swapped'])
    plot_tracking(a.out,{run:dfs[run]['sim_quality.csv'] for run in runs})
    plot_orientation(a.out,qs)
    txt='\n'.join(report)+'\n';(a.out/'report.txt').write_text(txt)
    print(txt);print('PLOTS AND REPORT:',a.out)

if __name__=='__main__':main()
