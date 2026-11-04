#!/usr/bin/env python3
"""Combine original SAM3+FoundationStereo wrist XYZ and stereo MediaPipe pose/fingers.

Wrist position = SAM3 masked FoundationStereo depth (visible wristband surface).
Palm orientation = triangulated 2-view MediaPipe landmarks from 12_stereo_g1_dex5.py.
Finger targets = 12_stereo_g1_dex5.py stereo retargeter (not re-estimated).
No synthetic orientation when palm reconstruction fails.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

FINGERS=('Pitch_', 'Roll_')
PALM_IDS=(0,5,9,17)

def normalize(vec):
    n=np.linalg.norm(vec)
    if not np.isfinite(n) or n<1e-8:raise ValueError('Degenerate palm')
    return vec/n

def palm_basis(points):
    w,i,m,p = points[list(PALM_IDS)]
    y=normalize(m-w)
    tmp=i-p
    x=normalize(tmp-np.dot(tmp,y)*y)
    z=normalize(np.cross(x,y))
    return np.column_stack((x,y,z))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True,help='Prepared wrist trial folder with wrist_xyz_camera1.csv and meta.json')
    p.add_argument('--stereo',type=Path,required=True,help='12_stereo_g1_dex5 output containing stereo_landmarks_and_targets.npz and stereo_quality.csv')
    p.add_argument('--table-frame',type=Path,required=True)
    args=p.parse_args()
    trial=args.trial.expanduser().resolve()
    stereo=args.stereo.expanduser().resolve()
    meta=json.loads((trial/'meta.json').read_text())
    take=Path(meta['take'])
    with (trial/'wrist_xyz_camera1.csv').open(newline='') as f:
        wrist=list(csv.DictReader(f))
    with (stereo/'stereo_quality.csv').open(newline='') as f:
        quality=list(csv.DictReader(f))
    with (take/'timestamps.csv').open(newline='') as f:
        timestamps={int(row['frame']):float(row.get('time_s') or row.get('cam1_retrieved_elapsed_s')) for row in csv.DictReader(f)}
    with np.load(args.table_frame) as z:
        Rt=np.asarray(z['R_cam1_to_table'],float)
        tt=np.asarray(z['t_cam1_to_table'],float).reshape(3)
    with np.load(stereo/'stereo_landmarks_and_targets.npz') as z:
        points=np.asarray(z['points_table_m'],float)
        valid=np.asarray(z['landmark_valid'],bool)
        group_valid=np.asarray(z['group_valid'],bool)
        names=[str(x) for x in z['group_names']]
    if 'wrist' not in names:raise RuntimeError('Missing wrist group validity in stereo NPZ')
    wrist_group=names.index('wrist')
    if not len(points)==len(quality)==len(valid)==len(group_valid):
        raise RuntimeError('Stereo quality/NPZ lengths mismatch')
    qmap={int(row['frame']):row for row in quality}
    keys=[k for k in quality[0] if k.startswith(FINGERS)]
    if len(keys)<8:raise RuntimeError('Too few Dex5 finger columns in stereo_quality.csv: '+str(keys))

    palm_path=trial/'palm_pose_6dof.csv'
    finger_path=trial/'joint_targets.csv'
    fields=['sequence_frame','source_video_frame','timestamp_s',
            'X_table_m','Y_table_m','Z_table_m',
            'palm_status','palm_qx_table','palm_qy_table','palm_qz_table','palm_qw_table',
            'wristband_depth_status','palm_orientation_method']
    npalm=0;nxyz=0
    with palm_path.open('w',newline='') as f1,finger_path.open('w',newline='') as f2:
        wp=csv.DictWriter(f1,fieldnames=fields);wp.writeheader()
        wf=csv.DictWriter(f2,fieldnames=['frame','thumb_state',*keys]);wf.writeheader()
        for row in wrist:
            source=int(row['source_video_frame']);seq=int(row['sequence_frame'])
            if source>=len(points):raise RuntimeError(f'Stereo tracker only has {len(points)} frames; needs source frame {source}')
            vals=np.array([float(row[k]) for k in ('X_cam1_m','Y_cam1_m','Z_cam1_m')])
            xyz=Rt@vals+tt if np.isfinite(vals).all() and row['status']=='VALID' else np.full(3,np.nan)
            if np.isfinite(xyz).all():nxyz+=1
            quat=[float('nan')]*4;st='INVALID'
            if bool(group_valid[source,wrist_group]) and valid[source,list(PALM_IDS)].all():
                try:
                    basis=palm_basis(points[source]);quat=Rotation.from_matrix(basis).as_quat()
                    st='VALID';npalm+=1
                except ValueError:pass
            wp.writerow(dict(zip(fields,[seq,source,timestamps.get(source,source/20),
                *xyz,st,*quat,row['status'],'stereo_triangulated_MediaPipe_PALM']))
            )
            # Source frame indexing is used to keep fingers matched with the two-camera recording.
            quality_row=qmap.get(source,{})
            wf.writerow({'frame':source,'thumb_state':quality_row.get('thumb_state','?'),
                         **{key:quality_row.get(key,'') for key in keys}})
    print('Palm CSV:',palm_path)
    print('Finger CSV:',finger_path)
    print(f'Wristband XYZ valid: {nxyz}/{len(wrist)}; stereo palm orientation valid: {npalm}/{len(wrist)}')
    print('Palm orientation is from stereo MediaPipe (NOT wristband and NOT FoundationStereo palm depth).')

if __name__=='__main__':main()
