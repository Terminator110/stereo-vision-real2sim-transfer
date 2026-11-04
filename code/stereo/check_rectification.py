#!/usr/bin/env python3
"""Stereo rectification viewer for horizontal OR vertical OpenCV stereo rigs.

Uses calibration/results/stereo_calibration.npz beside this script.
Controls: q quit, a cycle alpha 0/0.5/1, r raw/rectified, s save preview.
NOTE: Image 2 is NOT flipped; all processing uses raw camera frames.
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'calibration' / 'results' / 'stereo_calibration.npz'
CAM1, CAM2 = 0, 2
PREVIEW_WIDTH = 600

cal = np.load(DATA)
K1, D1, K2, D2 = (cal[x] for x in ('K1','D1','K2','D2'))
R, T = cal['R'], cal['T']
SIZE = tuple(int(x) for x in cal['image_size'])


def make_maps(alpha):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, SIZE, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    # Horizontal epipolar lines: translation occupies P2[0,3].
    # Vertical epipolar lines: translation occupies P2[1,3].
    axis = 'vertical' if abs(P2[1,3]) > abs(P2[0,3]) else 'horizontal'
    m1 = cv2.initUndistortRectifyMap(K1,D1,R1,P1,SIZE,cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K2,D2,R2,P2,SIZE,cv2.CV_32FC1)
    print(f'alpha={alpha}, rectification={axis}, P2 offset x={P2[0,3]:.1f}, y={P2[1,3]:.1f}')
    print('valid ROIs:', roi1, roi2)
    return m1,m2,axis


def open_cam(i):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open /dev/video{i}')
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w,h) != SIZE:
        cap.release()
        raise RuntimeError(f'/dev/video{i} opened at {w}x{h}, expected {SIZE}. Check USB speed.')
    return cap


def make_preview(img, label):
    h,w = img.shape[:2]
    scale = PREVIEW_WIDTH / w
    p = cv2.resize(img,(PREVIEW_WIDTH,round(h*scale)),interpolation=cv2.INTER_AREA)
    cv2.putText(p,label,(12,26),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,165,255),2)
    return p


def main():
    alpha_values = (0,0.5,1)
    alpha_index = 0
    maps1,maps2,axis = make_maps(alpha_values[alpha_index])
    c1 = c2 = None
    rectified = True
    try:
        c1, c2 = open_cam(CAM1),open_cam(CAM2)
        while True:
            if not c1.grab() or not c2.grab():
                print('Camera grab failed');break
            ok1,f1=c1.retrieve();ok2,f2=c2.retrieve()
            if not ok1 or not ok2:
                print('Camera retrieve failed');break
            if rectified:
                a = cv2.remap(f1,*maps1,cv2.INTER_LINEAR)
                b = cv2.remap(f2,*maps2,cv2.INTER_LINEAR)
            else:
                a,b=f1,f2
            p1=make_preview(a,'CAM 1')
            p2=make_preview(b,'CAM 2')
            combined=cv2.hconcat([p1,p2])
            if rectified:
                # Draw epipolar guides; vertical stereo => vertical lines.
                if axis=='vertical':
                    for x in range(0, PREVIEW_WIDTH,40):
                        cv2.line(combined,(x,0),(x,combined.shape[0]-1),(0,255,0),1)
                        cv2.line(combined,(x+PREVIEW_WIDTH,0),(x+PREVIEW_WIDTH,combined.shape[0]-1),(0,255,0),1)
                else:
                    for y in range(0, combined.shape[0],30):
                        cv2.line(combined,(0,y),(combined.shape[1]-1,y),(0,255,0),1)
            status=f"{'RECTIFIED' if rectified else 'RAW'} | alpha={alpha_values[alpha_index]} | {axis} stereo | a: alpha, r: raw, s: save, q: quit"
            cv2.putText(combined,status,(10,combined.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,0.44,(255,255,255),2)
            cv2.putText(combined,status,(10,combined.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,0.44,(0,0,0),1)
            cv2.imshow('Stereo Rectification Check',combined)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'):break
            if key==ord('r'):rectified=not rectified
            if key==ord('a'):
                alpha_index=(alpha_index+1)%len(alpha_values)
                maps1,maps2,axis=make_maps(alpha_values[alpha_index])
            if key==ord('s'):
                path=ROOT/'rectification_preview.png'
                cv2.imwrite(str(path),combined)
                print('Saved:',path)
    finally:
        if c1 is not None:c1.release()
        if c2 is not None:c2.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()
