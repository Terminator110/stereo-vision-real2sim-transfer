#!/usr/bin/env python3
"""Inverse-project MuJoCo yellow/actual IK world points onto original Camera 1 video.

WARNING: Requires verified *table origin translation*, yaw, and intentional workspace lift.
A round-trip of the yellow target is not independent validation of its stereo depth.
"""
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path.home()/'physikal_workflow/dex-5'

def parse():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--trial',type=Path,default=ROOT/'FoundationStereo/wristband_trial_new')
 p.add_argument('--sim',type=Path,default=ROOT/'outputs/dex5_sim_rest_start/sim_quality.csv')
 p.add_argument('--calib',type=Path,default=ROOT/'code/stereo/calibration/parallel_results_v3/stereo_calibration.npz')
 p.add_argument('--table',type=Path,default=ROOT/'code/stereo/calibration/parallel_results_v3/table_frame_multi.npz')
 p.add_argument('--table-origin-xy',nargs=2,type=float,default=[.375,-.045],metavar=('X','Y'))
 p.add_argument('--tabletop-world-z',type=float,default=.762)
 p.add_argument('--workspace-lift-z',type=float,default=.070,help='Intentional additional world Z on top of tabletop')
 p.add_argument('--yaw',type=float,default=-90.)
 p.add_argument('--frames',nargs='+',type=int,default=[10,60,100,120,149])
 p.add_argument('--out',type=Path,default=ROOT/'outputs/dex5_ik_video_overlay')
 p.add_argument('--video',action='store_true',help='Also render all matching frames to overlay_video.mp4')
 return p.parse_args()

def main():
 a=parse();a.out.mkdir(parents=True,exist_ok=True)
 with np.load(a.calib) as d: K=np.asarray(d['K1'],float);D=np.asarray(d['D1'],float)
 with np.load(a.table) as d: Rct=np.asarray(d['R_cam1_to_table'],float);tct=np.asarray(d['t_cam1_to_table'],float).reshape(3)
 phi=np.deg2rad(a.yaw);c,s=np.cos(phi),np.sin(phi)
 Rtw=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
 # Identical translation used for the configured XY registration, plus intentional lift.
 ttw=np.array([*a.table_origin_xy, a.tabletop_world_z+a.workspace_lift_z],float)
 with a.sim.open(newline='') as f: sim={int(r['sequence_frame']):r for r in csv.DictReader(f)}
 with (a.trial/'palm_pose_6dof.csv').open(newline='') as f: palm={int(r['sequence_frame']):r for r in csv.DictReader(f)}
 meta=json.loads((a.trial/'meta.json').read_text())
 take=Path(meta['take']).expanduser();take=take if take.is_absolute() else (a.trial/take).resolve()
 video=take/'camera1.avi';cap=cv2.VideoCapture(str(video))
 if not cap.isOpened():raise FileNotFoundError(f'Video not readable: {video}')
 def project_table(pt):
  cam=Rct.T@(pt-tct)
  if not np.isfinite(cam).all() or cam[2]<=0: return None
  uv,_=cv2.projectPoints(cam.reshape(1,1,3),np.zeros(3),np.zeros(3),K,D)
  return np.rint(uv.reshape(2)).astype(int)
 def marker(im,uv,col,label,shape):
  if uv is None:return
  u,v=map(int,uv)
  if shape=='cross':
   cv2.line(im,(u-13,v),(u+13,v),col,3,cv2.LINE_AA);cv2.line(im,(u,v-13),(u,v+13),col,3,cv2.LINE_AA)
  else:cv2.circle(im,(u,v),9,col,3,cv2.LINE_AA)
  cv2.putText(im,label,(u+14,v-12),cv2.FONT_HERSHEY_SIMPLEX,.57,col,2,cv2.LINE_AA)
 def get_point(r,prefix):return np.array([float(r[f'{prefix}_{ax}']) for ax in 'xyz'],float)
 def draw(seq):
  sr=sim[seq];pr=palm[seq]
  frame_idx=int(sr.get('source_video_frame',pr['source_video_frame']))
  cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
  ok,im=cap.read()
  if not ok:raise RuntimeError(f'Failed to read source frame {frame_idx}')
  human=np.array([float(pr[f'{ax}_table_m']) for ax in 'XYZ'],float)
  yellow=(Rtw.T@(get_point(sr,'wrist_target')-ttw))
  cyan=(Rtw.T@(get_point(sr,'wrist_actual')-ttw))
  uh=project_table(human);uy=project_table(yellow);uc=project_table(cyan)
  # Offset of the target after inverse registration is expected to be ~0 by construction.
  delta_px=np.linalg.norm(uh-uy) if uh is not None and uy is not None else np.nan
  marker(im,uh,(40,210,50),'Stereo wrist','cross')
  marker(im,uy,(0,220,255),'Yellow IK (inverse projection)','circle')
  marker(im,uc,(255,255,0),'Solved frame','cross')
  cv2.rectangle(im,(0,0),(min(im.shape[1],1120),114),(18,18,18),-1)
  texts=[f'Seq {seq} / source {frame_idx}   yaw {a.yaw:g} deg   intentional +Z {a.workspace_lift_z*1000:.0f}mm',
         f'Yellow vs stereo: {delta_px:.1f}px (round-trip consistency only)',
         f'IK pos error: {sr.get("position_error_mm","?")}mm  rot error: {sr.get("orientation_error_deg","?")}deg']
  for i,t in enumerate(texts):cv2.putText(im,t,(12,31+i*34),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),2,cv2.LINE_AA)
  return im,delta_px,uh,uy,uc
 frames=sorted(set(a.frames)&set(sim)&set(palm))
 if not frames:raise ValueError('No requested frames shared by CSVs')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(len(frames),1,figsize=(15,5*len(frames)),squeeze=False)
 for i,seq in enumerate(frames):
  im,delta,uh,uy,uc=draw(seq)
  axs[i,0].imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));axs[i,0].set_title(f'Frame {seq}: projected yellow vs reconstructed wrist (delta {delta:.1f} px)');axs[i,0].axis('off')
  cv2.imwrite(str(a.out/f'frame_{seq:03d}.png'),im)
 fig.tight_layout();fig.savefig(a.out/'ik_on_real_video_keyframes.png',dpi=130);plt.close(fig)
 print('Keyframes:',a.out/'ik_on_real_video_keyframes.png')
 if a.video:
  seqs=sorted(set(sim)&set(palm));writer=None
  for seq in seqs:
   im,*_=draw(seq)
   if writer is None:
    h,w=im.shape[:2];writer=cv2.VideoWriter(str(a.out/'ik_on_real_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(w,h))
    if not writer.isOpened():raise RuntimeError('Failed to open video writer')
   writer.write(im)
  if writer:writer.release()
  print('Overlay video:',a.out/'ik_on_real_video.mp4')
 cap.release()
 print('CAUTION: inverse-projected yellow is expected to overlay reconstructed wrist if transforms match; it does NOT validate anatomical wrist reference or depth.')
 print('Table-centre XY used as diagnostic origin; confirm actual table origin correspondence before interpreting absolute placement.')

if __name__=='__main__':main()
