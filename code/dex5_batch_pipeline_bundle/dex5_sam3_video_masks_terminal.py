#!/usr/bin/env python3
"""SAM3 video-propagated masks for rectified DEX5 wristband left_jpg frames."""
import argparse, csv
from pathlib import Path
import cv2
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--frames-dir',required=True,type=Path)
ap.add_argument('--weights',required=True,type=Path)
ap.add_argument('--out',required=True,type=Path)
ap.add_argument('--overlay-every',type=int,default=5)
ap.add_argument('--max-frames',type=int,default=0)
a=ap.parse_args()
files=sorted(a.frames_dir.glob('*.jpg'))
if a.max_frames: files=files[:a.max_frames]
if not files: ap.error('No JPGs found in '+str(a.frames_dir))
if not a.weights.is_file(): ap.error('SAM3 weights not found: '+str(a.weights))
a.out.mkdir(parents=True,exist_ok=True)
maskdir=a.out/'masks'; previewdir=a.out/'sam_overlays'
maskdir.mkdir(exist_ok=True); previewdir.mkdir(exist_ok=True)
first=cv2.imread(str(files[0]))
if first is None: ap.error('Cannot load '+str(files[0]))
h,w=first.shape[:2]
x,y,bw,bh=map(int,cv2.selectROI('Select ONLY wristband; Enter to confirm',first,False,False))
cv2.destroyAllWindows()
if bw<4 or bh<4: ap.error('No valid wristband ROI')
box=[x,y,x+bw,y+bh]
print('ROI:',box,'; frames:',len(files),flush=True)
video=a.out/'sam3_input.mp4'
writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(w,h))
if not writer.isOpened(): raise RuntimeError('Failed creating '+str(video))
for f in files:
    frame=cv2.imread(str(f))
    if frame is None or frame.shape[:2]!=(h,w): raise RuntimeError('Invalid '+str(f))
    writer.write(frame)
writer.release()

from ultralytics.models.sam import SAM3VideoPredictor
pred=SAM3VideoPredictor(overrides={
    'model':str(a.weights.resolve()),'task':'segment','mode':'predict',
    'imgsz':640,'conf':0.25,'quantize':16,'save':False,'verbose':False
})
prev=((box[0]+box[2])/2,(box[1]+box[3])/2)
reference_area=None
rows=[]
outputs=pred(source=str(video),bboxes=[box],stream=True)
for i,result in enumerate(outputs):
    if i>=len(files):break
    frame=cv2.imread(str(files[i]))
    best=None
    if result.masks is not None:
        for raw in result.masks.data.detach().cpu().numpy():
            if raw.shape!=(h,w):
                raw=cv2.resize(raw.astype('float32'),(w,h),interpolation=cv2.INTER_NEAREST)
            m=(raw>0.5).astype('uint8')
            yy,xx=np.nonzero(m)
            if len(xx)<20:continue
            center=(float(xx.mean()),float(yy.mean()))
            dist=np.hypot(center[0]-prev[0],center[1]-prev[1])
            if best is None or dist<best[0]:best=(dist,m,center,int(len(xx)))
    status='MISSING';m=np.zeros((h,w),dtype='uint8');area=0
    if best is not None:
        dist,candidate,center,area=best
        if i==0 or (dist<=80 and (reference_area is None or
                0.25*reference_area <= area <= 4.0*reference_area)):
            m=candidate*255;status='VALID';prev=center
            if reference_area is None:reference_area=area
        else: status='REJECTED_JUMP_OR_AREA'
    name=files[i].stem+'.png'
    cv2.imwrite(str(maskdir/name),m)
    if a.overlay_every>0 and i%a.overlay_every==0:
        vis=frame.copy();vis[m>0]=(
            .4*vis[m>0]+.6*np.array([0,255,0])
        ).astype('uint8')
        cv2.putText(vis,f'frame={i} {status} area={area}',(8,25),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
        cv2.imwrite(str(previewdir/name),vis)
    rows.append([i,name,status,int(np.count_nonzero(m))])
    if i%10==0:print(f'{i}/{len(files)} {status} pixels={rows[-1][-1]}',flush=True)
with (a.out/'sam_tracking.csv').open('w',newline='') as fh:
    cw=csv.writer(fh);cw.writerow(['frame','mask','status','mask_pixels']);cw.writerows(rows)
print('DONE:',len(rows),'/',len(files),'masks;',sum(r[2]=='VALID' for r in rows),'valid')
print('Inspect previews:',previewdir)
