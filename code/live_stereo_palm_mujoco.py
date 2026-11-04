#!/usr/bin/env python3
"""Live stereo palm in MuJoCo: default marker-only, no robot motion.
Requires code/12_stereo_g1_dex5.py and its sibling dependencies. Camera 2 is rotated
180 degrees after capture (live only). Both webcams must work.
"""
import argparse, importlib.util, time, csv
from pathlib import Path
import cv2, numpy as np, mujoco, mediapipe as mp
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path)); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod);return mod

def norm(v):
    m=np.linalg.norm(v)
    if not np.isfinite(m) or m<1e-7:raise ValueError('degenerate axis')
    return v/m

def palm(points):
    w,i,m,p=points[[0,5,9,17]]
    y=norm(m-w);x=norm((i-p)-y*np.dot(i-p,y));z=norm(np.cross(x,y))
    return np.column_stack((x,y,z))

def draw_axes(scene,origin,R,alpha=1.0,L=.085):
    for j,col in enumerate([(1.,.12,.12),(.1,1.,.1),(.15,.4,1.)]):
        if scene.ngeom >= scene.maxgeom:return
        g=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.full(3,.004),np.zeros(3),np.eye(3).ravel(),np.array([*col,alpha]))
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.004,np.asarray(origin,np.float64),np.asarray(origin+R[:,j]*L,np.float64))
        scene.ngeom+=1

def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cam1',type=int,default=2);p.add_argument('--cam2',type=int,default=0)
    p.add_argument('--rotate-cam2',action='store_true')
    p.add_argument('--calib',type=Path,default=Path('stereo/calibration/parallel_results_v3/stereo_calibration.npz'))
    p.add_argument('--table-frame',type=Path,default=Path('stereo/calibration/parallel_results_v3/table_frame_multi.npz'))
    p.add_argument('--hand-model',type=Path,default=Path('../models/mediapipe/hand_landmarker.task'))
    p.add_argument('--model',type=Path,default=Path('../models/scenario_g1_dex5_stationary.xml'))
    p.add_argument('--max-reproj',type=float,default=8.)
    p.add_argument('--max-grab-delta-ms',type=float,default=80.)
    p.add_argument('--table-yaw-deg',type=float,default=0.)
    p.add_argument('--palm-z-correction-deg',type=float,default=0.)
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--output',type=Path,default=Path('/tmp/live_palm_mujoco.csv'))
    a=p.parse_args()
    if a.cam1==a.cam2:p.error('Camera indices must differ')
    if not (HERE/'12_stereo_g1_dex5.py').is_file():p.error('Put this script in dex-5/code next to 12_stereo_g1_dex5.py')
    s=load('_live_stereo12',HERE/'12_stereo_g1_dex5.py')
    cal,size,Rtable,ttable=s.load_calibration(a.calib,a.table_frame)
    model=mujoco.MjModel.from_xml_path(str(a.model.resolve()));data=mujoco.MjData(model)
    data.qpos[:]=model.qpos0;mujoco.mj_forward(model,data)
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'g1_01_dex5_right_frame')
    if site<0:raise RuntimeError('Missing Dex5 site')
    # This geometric basis is a DIAGNOSTIC and must be validated visually.
    anchors=[]
    for n in ('Roll_21R','Roll_31R','Roll_51R'):
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'g1_01_dex5_right_'+n)
        if j<0:raise RuntimeError('Missing Dex5 knuckle joint: '+n)
        anchors.append(data.xanchor[j].copy())
    w=data.site_xpos[site].copy();di,dm,dp=anchors
    y=norm(dm-w);x=norm((di-dp)-y*np.dot(di-dp,y));z=norm(np.cross(x,y));dex_basis=np.column_stack((x,y,z))
    Rsite=data.site_xmat[site].reshape(3,3).copy(); site_to_anat=Rsite.T@dex_basis
    Rtw=Rotation.from_euler('z',a.table_yaw_deg,degrees=True).as_matrix()
    Rfix=Rotation.from_euler('z',a.palm_z_correction_deg,degrees=True).as_matrix()
    # MuJoCo robot is deliberately stationary, preventing unexpected table collisions.
    c1=cv2.VideoCapture(a.cam1);c2=cv2.VideoCapture(a.cam2)
    try:
        for c,n in [(c1,'Cam1'),(c2,'Cam2')]:
            if not c.isOpened():raise RuntimeError(n+' cannot open')
            c.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH,size[0]);c.set(cv2.CAP_PROP_FRAME_HEIGHT,size[1]);c.set(cv2.CAP_PROP_BUFFERSIZE,1)
        options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(a.hand_model.resolve())),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=1)
        from mujoco import viewer as mjviewer
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as h1, mp.tasks.vision.HandLandmarker.create_from_options(options) as h2, mjviewer.launch_passive(model,data) as viewer, a.output.open('w',newline='') as log:
            wr=csv.writer(log);wr.writerow(['frame','timestamp_s','status','grab_delta_ms','palm_change_from_first_deg','x_x','x_y','x_z','y_x','y_y','y_z','z_x','z_y','z_z'])
            viewer.cam.lookat[:]=data.site_xpos[site];viewer.cam.distance=1.15;viewer.cam.azimuth=135;viewer.cam.elevation=-22
            print('MARKER-ONLY: robot joints NEVER move. White/bright arrows at right = LIVE HUMAN; faint arrows at robot = FIXED DEX5 anatomical axes.')
            print('Red=X pinky->index; Green=Y wrist->middle MCP; Blue=Z palm normal. Q to quit. No first-frame alignment.')
            print('If Camera 2 is black, restore that feed first; no monocular fallback.')
            start=time.monotonic();last_ms=-1;first=None;good=0;n=0
            while viewer.is_running() and (a.max_frames<=0 or n<a.max_frames):
                ok1=c1.grab();t1=time.monotonic();ok2=c2.grab();t2=time.monotonic()
                if not (ok1 and ok2):print('Camera grab failed');break
                ok1,f1=c1.retrieve();ok2,f2=c2.retrieve()
                if not (ok1 and ok2):print('Camera retrieve failed');break
                if a.rotate_cam2:f2=cv2.rotate(f2,cv2.ROTATE_180)
                if (f1.shape[1],f1.shape[0])!=size or (f2.shape[1],f2.shape[0])!=size:raise RuntimeError('Camera size differs from calibration '+str(size))
                ms=max(last_ms+1,round((time.monotonic()-start)*1000));last_ms=ms
                img1=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(f1,cv2.COLOR_BGR2RGB)))
                img2=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(f2,cv2.COLOR_BGR2RGB)))
                l1,_=s.image_landmarks(h1.detect_for_video(img1,ms),size)
                l2,_=s.image_landmarks(h2.detect_for_video(img2,ms),size)
                delta=1000*(t2-t1)
                result=s.reject('GRAB_TIMING_REJECTED') if delta>a.max_grab_delta_ms else s.reconstruct_hand(l1,l2,cal,Rtable,ttable,a.max_reproj)
                H=None;deg=float('nan')
                if result['points_table'] is not None and result['landmark_valid'][[0,5,9,17]].all() and result['groups'].get('wrist'):
                    H=Rtw@palm(result['points_table']);H=H@Rfix
                    if first is None:first=H.copy()
                    deg=np.degrees(Rotation.from_matrix(H@first.T).magnitude());good+=1
                with viewer.lock():
                    viewer.user_scn.ngeom=0
                    draw_axes(viewer.user_scn,data.site_xpos[site],dex_basis,alpha=.35)
                    if H is not None:draw_axes(viewer.user_scn,data.site_xpos[site]+np.array([0,-.20,.14]),H,alpha=1.)
                viewer.sync()
                overlay=f'{result["status"]} | valid palms {good}/{n+1} | rotation since first {deg:.1f} deg | q=quit'
                cv2.putText(f1,overlay,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                cv2.putText(f2,'CAM2 (rotated if requested)',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                cv2.imshow('CAM1 / original source',f1);cv2.imshow('CAM2 / stereo source',f2)
                if cv2.waitKey(1)&255 in (ord('q'),27):break
                arr=H if H is not None else np.full((3,3),np.nan)
                wr.writerow([n,ms/1000,result['status'],delta,deg,*arr[:,0],*arr[:,1],*arr[:,2]])
                if n%20==0:print(f'{n}: {result["status"]}; palm {good}/{n+1}; delta={deg:.1f}deg',flush=True)
                n+=1
            print(f'Done: valid palm orientations {good}/{n}; CSV {a.output}')
    finally:
        c1.release();c2.release();cv2.destroyAllWindows()
if __name__=='__main__':run()
