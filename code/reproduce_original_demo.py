#!/usr/bin/env python3
"""Reproduce the retained stereo pipeline without additional wrist-depth gates."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from reconstruct_demo import CODE, ROOT, CALIB, TABLE, REFERENCE, combine, take_info, load_csv, write_csv, sha

NAME='take_20261103_211944_200460'
TAKE=CODE/'stereo/recordings_parallel_v3'/NAME
PREPARED=ROOT/'FoundationStereo'/('wristband_batch_'+NAME)
PREVIOUS=ROOT/'outputs/reconstructed_demos'/NAME


def execute(command, logfile):
    print('$ '+shlex.join(list(map(str,command))),flush=True)
    with logfile.open('w') as log:
        process=subprocess.Popen(list(map(str,command)),cwd=CODE,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,
                                 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        try:
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if code:raise RuntimeError(f'Command failed ({code}); see {logfile}')


def scene_copy(output):
    original=PREVIOUS/'scenario_recorded.xml'
    tree=ET.parse(original)
    original_scene=PREVIOUS/'recorded_scene_objects.xml'
    copied_scene=output/'recorded_scene_objects.xml'
    shutil.copy2(original_scene,copied_scene)
    replaced=0
    for node in tree.findall('include'):
        if Path(node.get('file')).resolve()==original_scene.resolve():
            node.set('file',str(copied_scene));replaced+=1
    if replaced!=1:raise ValueError('Could not identify original per-take objects include')
    target=output/'scenario_recorded.xml';tree.write(target,encoding='unicode')
    a=mujoco.MjModel.from_xml_path(str(original));b=mujoco.MjModel.from_xml_path(str(target))
    for key in ('body_pos','body_quat','geom_pos','geom_quat','cam_pos','cam_quat','qpos0','jnt_range'):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    return target


def intervals(frames):
    runs=[]
    for frame in frames:
        if runs and frame==runs[-1][-1]+1:runs[-1].append(frame)
        else:runs.append([frame])
    return [dict(first=r[0],last=r[-1],length=len(r))for r in runs]


def analyse(output,protected):
    wrist=load_csv(output/'wrist_xyz_camera1.csv')
    previous=load_csv(PREVIOUS/'trial/wrist_xyz_camera1.csv')
    report_depth=dict(total=len(wrist),valid=sum(r['status']=='VALID'for r in wrist))
    report_depth['invalid']=len(wrist)-report_depth['valid']
    report_depth['invalid_intervals']=intervals([int(r['source_video_frame'])for r in wrist if r['status']!='VALID'])
    extra=[dict(frame=int(a['source_video_frame']),previous_rejection=b['status'])for a,b in zip(wrist,previous)
           if a['status']=='VALID'and b['status']!='VALID']
    report_depth.update(previous_agent_additional_rejections=extra,
                        additional_rejection_count=len(extra),additional_filtering_applied=False)
    (output/'wrist_validity_report.json').write_text(json.dumps(report_depth,indent=2)+'\n')
    if (len(wrist),report_depth['valid'],report_depth['invalid'])!=(186,157,29):
        raise ValueError('Source wrist data no longer matches the verified 157/186 result')
    new=load_csv(output/'joint_targets.csv');reference=load_csv(REFERENCE/'joint_targets.csv')
    original_quality=load_csv(output/'stereo_retarget/stereo_quality.csv')
    if list(new[0])!=list(reference[0]):raise ValueError('Finger CSV headers differ from successful reference')
    model=mujoco.MjModel.from_xml_path(str(output/'scenario_recorded.xml'))
    saved=np.load(output/'simulation/sim_qpos.npz')
    np.testing.assert_array_equal(saved['source_frames'],np.arange(186))
    statistics=[]
    for key in reference[0]:
        if not key.startswith(('Yaw_','Roll_','Pitch_')):continue
        joint=model.joint('g1_01_dex5_right_'+key);address=int(model.jnt_qposadr[joint.id])
        values=np.array([float(r[key])for r in new]);baseline=np.array([float(r[key])for r in reference])
        np.testing.assert_array_equal(values,np.array([float(r[key])for r in original_quality]))
        actual=saved['qpos'][:,address]
        expected=np.clip(values,*model.jnt_range[joint.id])
        np.testing.assert_allclose(actual,expected,atol=1e-10)
        statistics.append(dict(joint=key,qpos_address=address,reference_range_rad=float(np.ptp(baseline)),
                               new_source_range_rad=float(np.ptp(values)),replay_range_rad=float(np.ptp(actual)),
                               maximum_mapping_error_rad=float(np.max(np.abs(actual-expected)))))
    if len(statistics)!=20:raise ValueError('Expected all 20 Dex5 finger joints')
    write_csv(output/'finger_movement_statistics.csv',statistics)
    quality=load_csv(output/'simulation/sim_quality.csv')
    arm_addresses=np.array([int(model.jnt_qposadr[model.joint('g1_01_right_'+name+'_joint').id])
                            for name in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')])
    invalid=np.array([r['status']!='VALID'for r in wrist])
    for i in np.flatnonzero(invalid):
        previous_q=saved['qpos'][i-1]if i else model.qpos0
        np.testing.assert_array_equal(saved['qpos'][i,arm_addresses],previous_q[arm_addresses])
    invalid_updates=[int(r['source_video_frame'])for r in quality
                     if r['wrist_xyz_valid']=='False'and int(r['finger_joints_applied'])==20]
    if len(invalid_updates)!=29:raise ValueError('Finger updates were lost during missing wrist depth')
    changed_during_invalid=sum(bool(np.any(np.abs(saved['qpos'][i,[r['qpos_address']for r in statistics]]-
                    saved['qpos'][i-1,[r['qpos_address']for r in statistics]])>1e-8))for i in np.flatnonzero(invalid)if i)
    palms=load_csv(output/'palm_pose_6dof.csv')
    if list(palms[0])!=list(load_csv(REFERENCE/'palm_pose_6dof.csv')[0]):raise ValueError('Palm header mismatch')
    basis_spec=importlib.util.spec_from_file_location('existing_basis',CODE/'combine_wristband_stereo_pose.py')
    basis=importlib.util.module_from_spec(basis_spec);basis_spec.loader.exec_module(basis)
    with np.load(output/'stereo_retarget/stereo_landmarks_and_targets.npz')as z:
        for i,row in enumerate(palms):
            if row['palm_status']=='VALID':
                quaternion=np.array([float(row[k])for k in ('palm_qx_table','palm_qy_table','palm_qz_table','palm_qw_table')])
                expected=Rotation.from_matrix(basis.palm_basis(z['points_table_m'][i])).as_matrix()
                np.testing.assert_allclose(Rotation.from_quat(quaternion).as_matrix(),expected,atol=1e-12)
    unchanged=all(sha(Path(path))==checksum for path,checksum in protected.items())
    if not unchanged:raise ValueError('A protected source artifact changed')
    active=[r for r in quality if r['orientation_ik_active']=='True']
    summary=dict(frames=186,wrist=report_depth,complete_stereo_frames=sum(r['stereo_status']=='STEREO_VALID'for r in original_quality),
                 palm_orientation_valid=sum(r['palm_status']=='VALID'for r in palms),
                 mapped_finger_joints=len(statistics),finger_updates_during_missing_xyz=len(invalid_updates),
                 invalid_xyz_frames_with_actual_finger_motion=changed_during_invalid,
                 original_csv_headers_match=True,palm_convention_verified=True,protected_files_unchanged=unchanged,
                 simulation=json.loads((output/'simulation/sim_report.json').read_text()),
                 active_orientation_frames=len(active),
                 active_orientation_median_error_deg=float(np.median([float(r['orientation_error_deg'])for r in active]))if active else None,
                 warning='No extra depth rejection; original VALID samples may still contain outliers. Kinematic replay is not verified physical grasp or collision avoidance.')
    (output/'reconstruction_quality_report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:summary[key]for key in ('frames','mapped_finger_joints','finger_updates_during_missing_xyz','invalid_xyz_frames_with_actual_finger_motion')},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='NEW directory; existing paths are refused')
    parser.add_argument('--verify-only',action='store_true',help='Validate an already generated directory without repeating reconstruction')
    args=parser.parse_args();output=args.output.expanduser().resolve()
    if args.verify_only:
        protected=json.loads((output/'provenance/protected_hashes.json').read_text())
        analyse(output,protected)
        return
    output.mkdir(parents=True,exist_ok=False)
    (output/'logs').mkdir();(output/'provenance').mkdir()
    protected_paths=[CALIB,TABLE,PREPARED/'wrist_xyz_camera1.csv',ROOT/'models/scenario_g1_dex5_stationary.xml',
                     *[REFERENCE/name for name in ('palm_pose_6dof.csv','joint_targets.csv','wrist_xyz_camera1.csv')],
                     *[PREVIOUS/name for name in ('scenario_recorded.xml','recorded_scene_objects.xml','simulation/sim_qpos.npz','trial/palm_pose_6dof.csv','trial/joint_targets.csv')]]
    protected={str(path):sha(path)for path in protected_paths}
    (output/'provenance/protected_hashes.json').write_text(json.dumps(protected,indent=2)+'\n')
    for name in ('dex5_video_to_sim_wrist_smoothed.py','12_stereo_g1_dex5.py','combine_wristband_stereo_pose.py'):
        shutil.copy2(CODE/name,output/'provenance'/name)
    _,n,times=take_info(TAKE)
    for name in ('meta.json','wrist_xyz_camera1.csv'):
        shutil.copy2(PREPARED/name,output/name)
    execute([sys.executable,CODE/'12_stereo_g1_dex5.py','--left-video',TAKE/'camera1.avi','--right-video',TAKE/'camera2.avi',
             '--calib',CALIB,'--table-frame',TABLE,'--max-reproj','8','--headless','--output',output/'stereo_retarget'],output/'logs/stereo.log')
    combine(output,output/'stereo_retarget',times,n)
    # Match the successful reference schema exactly, without losing Yaw_11R.
    for name in ('palm_pose_6dof.csv','joint_targets.csv'):
        rows=load_csv(output/name);fields=list(load_csv(REFERENCE/name)[0])
        write_csv(output/name,[{key:row[key]for key in fields}for row in rows],fields)
    scenario=scene_copy(output)
    replay=[sys.executable,CODE/'dex5_video_to_sim_wrist_smoothed.py','--palm-csv',output/'palm_pose_6dof.csv',
            '--finger-csv',output/'joint_targets.csv','--model',scenario,'--video',TAKE/'camera1.avi',
            '--ik-reference','wrist','--ik-mode','pose','--table-yaw-deg','-90',
            '--table-origin-world','.385','-.120','.832','--xyz-window','9','--orientation-window','7',
            '--orientation-max-gap-s','.30','--position-weight','10','--pose-weight','.5',
            '--max-arm-speed','4','--preposition-seconds','2','--speed','1','--output',output/'simulation']
    execute([*replay,'--headless'],output/'logs/replay.log')
    (output/'replay.sh').write_text('#!/usr/bin/env bash\nset -e\nexport MUJOCO_GL=glfw\nexec '+shlex.join(list(map(str,replay)))+' "$@"\n')
    analyse(output,protected)
    print('Review replay: bash',output/'replay.sh')


if __name__=='__main__':main()
