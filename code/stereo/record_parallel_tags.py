#!/usr/bin/env python3
"""Record both stereo cameras and triangulated AprilTags in one indexed take."""
import argparse, csv, datetime, time, pathlib
import cv2, numpy as np

def camera(idx,w,h,fps):
    c=cv2.VideoCapture(idx,cv2.CAP_V4L2)
    if not c.isOpened(): raise RuntimeError(f'Cannot open /dev/video{idx}')
    c.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    c.set(cv2.CAP_PROP_FRAME_WIDTH,w);c.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
    c.set(cv2.CAP_PROP_FPS,fps); c.set(cv2.CAP_PROP_BUFFERSIZE,1)
    size=(round(c.get(3)),round(c.get(4)))
    if size!=(w,h): raise RuntimeError(f'Camera {idx} gives {size}, requested {(w,h)}')
    return c

def marker_centres(frame,detector,ids_wanted):
    corners,ids,_=detector.detectMarkers(frame)
    found={}
    if ids is not None:
        for poly,tag in zip(corners,ids.ravel()):
            if int(tag) in ids_wanted: found[int(tag)]=np.asarray(poly,dtype=np.float64).reshape(4,2).mean(axis=0)
    return found

def triangulate(p1,p2,K1,D1,K2,D2,R,T):
    a=cv2.undistortPoints(np.asarray(p1,np.float64).reshape(1,1,2),K1,D1).reshape(2,1)
    b=cv2.undistortPoints(np.asarray(p2,np.float64).reshape(1,1,2),K2,D2).reshape(2,1)
    P1=np.column_stack([np.eye(3),np.zeros((3,1))]); P2=np.column_stack([R,T.reshape(3,1)])
    Xh=cv2.triangulatePoints(P1,P2,a,b).reshape(4)
    if abs(Xh[3])<1e-10: return None,None
    X=Xh[:3]/Xh[3]
    X2=R@X+T.reshape(3)
    if X[2]<=0 or X2[2]<=0: return None,None
    q1=cv2.projectPoints(X.reshape(1,1,3),np.zeros(3),np.zeros(3),K1,D1)[0].reshape(2)
    rvec=cv2.Rodrigues(R)[0]
    q2=cv2.projectPoints(X.reshape(1,1,3),rvec,T.reshape(3),K2,D2)[0].reshape(2)
    err=float(np.sqrt((np.sum((q1-p1)**2)+np.sum((q2-p2)**2))/2))
    return X,err

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cam1',type=int,default=2);ap.add_argument('--cam2',type=int,default=0)
    ap.add_argument('--rotate-cam2',action='store_true')
    ap.add_argument('--calib',default='stereo/calibration/parallel_results/stereo_calibration.npz')
    ap.add_argument('--ids',nargs='+',type=int,default=[2,4,5]);ap.add_argument('--out-root',default='stereo/recordings_parallel')
    ap.add_argument('--width',type=int,default=1280);ap.add_argument('--height',type=int,default=720)
    ap.add_argument('--fps',type=float,default=20.);ap.add_argument('--max-reproj',type=float,default=3.)
    a=ap.parse_args(); d=np.load(a.calib)
    K1,D1,K2,D2,R,T=[d[k] for k in ['K1','D1','K2','D2','R','T']]
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),cv2.aruco.DetectorParameters())
    root=pathlib.Path(a.out_root);root.mkdir(parents=True,exist_ok=True)
    take=root/('take_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));take.mkdir()
    cap1=camera(a.cam1,a.width,a.height,30);cap2=camera(a.cam2,a.width,a.height,30)
    fourcc=cv2.VideoWriter_fourcc(*'MJPG'); size=(a.width,a.height)
    w1=cv2.VideoWriter(str(take/'camera1.avi'),fourcc,a.fps,size)
    w2=cv2.VideoWriter(str(take/'camera2.avi'),fourcc,a.fps,size)
    if not w1.isOpened() or not w2.isOpened(): raise RuntimeError('VideoWriter failed')
    print('Saving to',take,flush=True)
    print('Press Q in the preview window to stop. Camera 2 rotation is saved in camera2.avi.',flush=True)
    start=time.monotonic();i=0
    try:
        with (take/'tags_xyz.csv').open('w',newline='') as fh,(take/'timestamps.csv').open('w',newline='') as tf:
            csvw=csv.writer(fh);tw=csv.writer(tf)
            csvw.writerow(['frame','time_s','tag_id','x_cam1_m','y_cam1_m','z_cam1_m','reproj_px','cam1_seen','cam2_seen','valid'])
            tw.writerow(['frame','time_s','cam1_monotonic_ns','cam2_monotonic_ns'])
            while True:
                if not(cap1.grab() and cap2.grab()): print('Camera grab failed');break
                ok1,f1=cap1.retrieve();t1=time.monotonic_ns()
                ok2,f2=cap2.retrieve();t2=time.monotonic_ns()
                if not(ok1 and ok2):break
                if a.rotate_cam2:f2=cv2.rotate(f2,cv2.ROTATE_180)
                elapsed=(time.monotonic()-start)
                w1.write(f1);w2.write(f2);tw.writerow([i,f'{elapsed:.9f}',t1,t2])
                obs1=marker_centres(f1,detector,set(a.ids));obs2=marker_centres(f2,detector,set(a.ids))
                valid_count=0
                for tag in a.ids:
                    x=y=z=err=float('nan');valid=0
                    if tag in obs1 and tag in obs2:
                        pt,err=triangulate(obs1[tag],obs2[tag],K1,D1,K2,D2,R,T)
                        if pt is not None and err<=a.max_reproj:
                            x,y,z=map(float,pt);valid=1;valid_count+=1
                    csvw.writerow([i,f'{elapsed:.9f}',tag,x,y,z,err,int(tag in obs1),int(tag in obs2),valid])
                if i%20==0:fh.flush();tf.flush()
                panel=cv2.hconcat([cv2.resize(f1,(640,360)),cv2.resize(f2,(640,360))])
                cv2.putText(panel,f'REC {elapsed:.1f}s frame {i} 3D tags {valid_count}',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
                cv2.imshow('Parallel stereo + AprilTag recording (Q stops)',panel)
                i+=1
                if cv2.waitKey(1)&0xff in (ord('q'),27):break
    finally:
        for obj in (cap1,cap2,w1,w2):obj.release()
        cv2.destroyAllWindows()
    print(f'Saved {i} paired frames to {take}',flush=True)
    print('WARNING: encoded playback FPS is fixed; timestamps.csv contains acquisition timing.',flush=True)
if __name__=='__main__':main()
