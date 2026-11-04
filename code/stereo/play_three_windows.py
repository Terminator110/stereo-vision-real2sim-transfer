#!/usr/bin/env python3
"""Play aligned camera AVI frames + 3D tag positions in three GUI windows; terminal P to start."""
import argparse,csv,threading,time,sys,pathlib
from collections import defaultdict
import cv2,numpy as np
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--take',required=True)
    ap.add_argument('--fps',type=float,default=0,help='Override playback FPS; otherwise use AVI FPS')
    ap.add_argument('--show-all',action='store_true',help='Show past and future samples (default past only)')
    a=ap.parse_args();take=pathlib.Path(a.take)
    v1=cv2.VideoCapture(str(take/'camera1.avi'));v2=cv2.VideoCapture(str(take/'camera2.avi'))
    if not (v1.isOpened() and v2.isOpened()):raise SystemExit('Missing camera1.avi or camera2.avi')
    csvpath=take/'tags_xyz.csv'
    if not csvpath.exists():raise SystemExit(f'Missing {csvpath}; use integrated recorder for matching frames.')
    samples=defaultdict(list)
    with csvpath.open(newline='') as fh:
        for row in csv.DictReader(fh):
            try:
                if int(row['valid'])!=1:continue
                frame=int(row['frame']);tag=int(row['tag_id'])
                p=np.array([float(row[k]) for k in ('x_cam1_m','y_cam1_m','z_cam1_m')])*1000
                if np.isfinite(p).all():samples[tag].append((frame,p))
            except (KeyError,ValueError):continue
    if not samples: raise SystemExit('No valid 3D positions in tags_xyz.csv')
    total=min(int(v1.get(cv2.CAP_PROP_FRAME_COUNT)),int(v2.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps=a.fps or v1.get(cv2.CAP_PROP_FPS) or 20
    fig=plt.figure('3D tag reconstruction (Camera 1 coordinates)',figsize=(7,6))
    ax=fig.add_subplot(111,projection='3d');colors={2:'tab:orange',4:'tab:blue',5:'tab:green'}
    points=np.concatenate([np.stack([p for _,p in vals]) for vals in samples.values()])
    lo=points.min(axis=0);hi=points.max(axis=0);ctr=(lo+hi)/2;span=max(float(np.ptp(points,axis=0).max()),100.)*.62
    ax.set(xlim=(ctr[0]-span,ctr[0]+span),ylim=(ctr[1]-span,ctr[1]+span),zlim=(ctr[2]-span,ctr[2]+span),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z / depth (mm)')
    tracks={};dots={}
    for tag, vals in sorted(samples.items()):
        color=colors.get(tag,'tab:red')
        tracks[tag],=ax.plot([],[],[],color=color,lw=1.6,label=f'Tag {tag}')
        dots[tag],=ax.plot([],[],[],'o',color=color,ms=8)
    ax.legend();fig.canvas.draw();plt.show(block=False)
    cv2.namedWindow('Camera 1 - recorded',cv2.WINDOW_NORMAL)
    cv2.namedWindow('Camera 2 - recorded',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera 1 - recorded',800,450);cv2.resizeWindow('Camera 2 - recorded',800,450)
    go=threading.Event()
    def read_terminal():
        print('\nThree windows open. TYPE p THEN ENTER HERE to begin. q + Enter to quit.',flush=True)
        for line in sys.stdin:
            cmd=line.strip().lower()
            if cmd=='p':go.set();return
            if cmd=='q':return
    th=threading.Thread(target=read_terminal,daemon=True);th.start()
    print(f'Loaded {total} video frames at playback {fps:.2f} FPS, tags {sorted(samples)}',flush=True)
    try:
        while not go.is_set() and th.is_alive():
            plt.pause(.05)
            if cv2.waitKey(1)&0xff in (ord('q'),27):return
        if not go.is_set():return
        i=0;clock=time.monotonic()
        while i<total:
            ok1,f1=v1.read();ok2,f2=v2.read()
            if not(ok1 and ok2):break
            cv2.imshow('Camera 1 - recorded',f1);cv2.imshow('Camera 2 - recorded',f2)
            for tag,vals in samples.items():
                subset=[p for idx,p in vals if idx<=i or a.show_all]
                if subset:
                    xyz=np.stack(subset)
                    tracks[tag].set_data(xyz[:,0],xyz[:,1]);tracks[tag].set_3d_properties(xyz[:,2])
                    at=[p for idx,p in vals if idx==i]
                    if at:
                        dots[tag].set_data([at[-1][0]],[at[-1][1]]);dots[tag].set_3d_properties([at[-1][2]])
                    else:
                        dots[tag].set_data([],[]);dots[tag].set_3d_properties([])
                else:
                    tracks[tag].set_data([],[]);tracks[tag].set_3d_properties([])
                    dots[tag].set_data([],[]);dots[tag].set_3d_properties([])
            ax.set_title(f'Frame {i}/{total-1}  |  t={i/fps:.2f}s')
            fig.canvas.draw_idle();fig.canvas.flush_events()
            if cv2.waitKey(1)&0xff in (ord('q'),27):break
            i+=1
            # Matplotlib drawing may be slower than camera playback; keep all frames,
            # with wall time permitted to lag rather than deliberately skip frames.
            remain=clock+i/fps-time.monotonic()
            if remain>0:time.sleep(remain)
    finally:
        v1.release();v2.release();cv2.destroyAllWindows();plt.close(fig)
    print('Playback complete.',flush=True)
if __name__=='__main__':main()
