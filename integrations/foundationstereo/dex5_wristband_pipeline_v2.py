#!/usr/bin/env python3
"""DEX-5 wristband 3D tracking: prepare -> external SAM3 tracker -> FoundationStereo depth.

Runs against the existing v3 stereo calibration and recordings. Input recordings
are already upright (Camera 2 was rotated on capture).

Coordinate frames:
  FS: the rotated, rectified Camera 1 frame used by FoundationStereo.
  rect_cam1: OpenCV rectified Camera 1 (before 90 deg CCW image rotation).
  cam1: ORIGINAL Camera 1 optical frame, in metres. NO table frame mapping.

Outputs 3D centroids of the VISIBLE WRISTBAND SURFACE, not anatomical wrist joint.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import types

import cv2
import numpy as np

PROJECT = Path.home() / 'physikal_workflow/dex-5'
STEREO = PROJECT / 'code/stereo'
DEFAULT_CALIB = STEREO / 'calibration/parallel_results_v3/stereo_calibration.npz'
DEFAULT_TAKE = STEREO / 'recordings_parallel_v3/take_20261102_165150_451555'
DEFAULT_OUT = PROJECT / 'FoundationStereo/wristband_v3'
DEFAULT_FS = PROJECT / 'FoundationStereo'


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def get_frame(cap, ix):
    cap.set(cv2.CAP_PROP_POS_FRAMES, ix)
    ok, img = cap.read()
    if not ok:
        raise RuntimeError(f'Unable to read frame {ix}')
    return img


def prepare(args):
    out = args.out.resolve()
    left_png = out / 'left_png'
    right_png = out / 'right_png'
    left_jpg = out / 'left_jpg'
    for d in (left_png, right_png, left_jpg):
        d.mkdir(parents=True, exist_ok=True)

    with np.load(args.calib) as c:
        K1, D1, R1, P1 = (c[k] for k in ('K1','D1','R1','P1'))
        K2, D2, R2, P2 = (c[k] for k in ('K2','D2','R2','P2'))
        W, H = [int(v) for v in c['image_size']]
        baseline_m = float(c['baseline_mm']) / 1000
        baseline_from_P2 = -float(P2[1, 3]) / float(P2[1, 1])
    if abs(float(P2[0, 3])) > 1e-4 or float(P2[1, 3]) >= 0:
        raise ValueError('Expected positive vertical disparity: check v3 camera orientation / P2')
    if abs(baseline_from_P2 - baseline_m) > 1e-4:
        raise ValueError('Projection and calibration baseline disagree')
    if not 0 < args.scale <= 1:
        raise ValueError('--scale must be in (0,1]')

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1[:, :3], (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2[:, :3], (W, H), cv2.CV_32FC1)

    # Original rectified image coordinates u,v -> rotated CCW u'=v, v'=W-1-u
    # Calibrated vertical d = yL - yR >0 becomes horizontal d = xL - xR >0.
    K_fs = np.array([[P1[1, 1] * args.scale, 0, P1[1, 2] * args.scale],
                     [0, P1[0, 0] * args.scale, (W - 1 - P1[0, 2]) * args.scale],
                     [0, 0, 1.0]], dtype=np.float64)

    lvid = args.take / 'camera1.avi'
    rvid = args.take / 'camera2.avi'
    cap1 = cv2.VideoCapture(str(lvid))
    cap2 = cv2.VideoCapture(str(rvid))
    if not cap1.isOpened() or not cap2.isOpened():
        raise FileNotFoundError(f'Missing/open failed: {lvid} / {rvid}')
    n = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    if args.end == -1:
        end = n - 1
    else:
        end = min(args.end, n - 1)
    if args.start < 0 or args.start > end or args.step < 1:
        raise ValueError(f'Bad start/end/step, video has {n} frames')

    selected = list(range(args.start, end + 1, args.step))
    if args.max_frames > 0:
        selected = selected[:args.max_frames]
    # Numeric video frame names are consecutively indexed 00000, 00001, ...
    for j, src_ix in enumerate(selected):
        img1 = get_frame(cap1, src_ix)
        img2 = get_frame(cap2, src_ix)
        if img1.shape[:2] != (H, W) or img2.shape[:2] != (H, W):
            raise RuntimeError('Recorded frame size does not match v3 calibration')
        rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
        a = cv2.rotate(rect1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        b = cv2.rotate(rect2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Preserve full rotated field of view; resizing does not crop it.
        if args.scale != 1.0:
            a = cv2.resize(a, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        name = f'{j:05d}'
        if not cv2.imwrite(str(left_png / (name + '.png')), a):
            raise IOError('Failed saving left PNG')
        if not cv2.imwrite(str(right_png / (name + '.png')), b):
            raise IOError('Failed saving right PNG')
        if not cv2.imwrite(str(left_jpg / (name + '.jpg')), a, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise IOError('Failed saving SAM3 left JPG')
        if j % 20 == 0 or j == len(selected)-1:
            print(f'Prepared {j+1}/{len(selected)} (source frame {src_ix})', flush=True)
    cap1.release(); cap2.release()

    meta = {
        'take': str(args.take.resolve()), 'calib': str(args.calib.resolve()),
        'source_video_frames': selected, 'video_fps_reported': float(fps1),
        'frame_time_note': 'Video FPS may be inaccurate; use original timestamps.csv for real timing.',
        'source_timestamps_csv': str(args.take / 'timestamps.csv'),
        'width': int(a.shape[1]), 'height': int(a.shape[0]),
        'scale': float(args.scale), 'K_fs': K_fs.tolist(), 'R1': R1.tolist(),
        'baseline_m': baseline_m,
        'coordinates': 'K_fs is for CCW-rotated and downscaled rectified LEFT image. '
                       'R1 maps Camera1 original to rectified camera1.',
    }
    save_json(out / 'meta.json', meta)
    (out / 'K.txt').write_text(' '.join(f'{v:.12f}' for v in K_fs.ravel()) + f'\n{baseline_m:.12f}\n')
    # visual check: same-height horizontal scanlines in both rectified, rotated images
    a = cv2.imread(str(left_png / '00000.png'))
    b = cv2.imread(str(right_png / '00000.png'))
    preview = np.concatenate([a, b], axis=1)
    for y in range(0, preview.shape[0], 70):
        cv2.line(preview, (0,y), (preview.shape[1]-1,y), (0,255,0), 1)
    cv2.imwrite(str(out / 'epipolar_preview.png'), preview)
    print('\nPrepared', len(selected), 'stereo pairs at', a.shape[:2], '(height,width)')
    print('Output:', out)
    print('Prompt image:', left_jpg / '00000.jpg')
    print('Stereo baseline:', baseline_m, 'm; horizontal fx:', K_fs[0,0])
    print('No anatomical wrist centre or table coordinate conversion is implied.')


def load_fs(args):
    # In NVIDIA's original Utils.py Open3D is imported at module import time,
    # though depth-only FoundationStereo inference does not invoke Open3D.
    # This stub avoids heavyweight visualization dependencies; NO point-cloud calls.
    sys.modules['open3d'] = types.ModuleType('open3d')
    sys.path.insert(0, str(args.fs_repo.resolve()))
    import torch
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    ckpt = args.fs_ckpt.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    cfg = OmegaConf.load(str(ckpt.parent / 'cfg.yaml'))
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    cfg['valid_iters'] = args.iters
    cfg['scale'] = 1.0  # already scaled in the prepare stage
    cfg['hiera'] = 0
    cfg['get_pc'] = 0
    print('Loading FoundationStereo checkpoint:', ckpt, flush=True)
    model = FoundationStereo(cfg)
    # Trusted NVIDIA checkpoint only. torch 2.7 otherwise defaults weights_only=True.
    ckpt_data = torch.load(str(ckpt), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt_data['model'])
    model.cuda().eval()
    return model, InputPadder, torch


def sample_band_xyz(mask, disp, K_fs, R1, baseline, zmin, zmax, erode_px, min_pts):
    h,w = disp.shape
    if mask.shape != (h,w):
        raise ValueError('Mask/disparity must have identical pixel dimensions')
    m = (mask > 0).astype(np.uint8)
    total = int(m.sum())
    if erode_px > 1:
        kernel = np.ones((erode_px,erode_px), np.uint8)
        m = cv2.erode(m, kernel, iterations=1)
    if int(m.sum()) == 0:
        return None, total, 0, np.nan, np.nan
    yy,xx = np.indices((h,w), dtype=np.float32)
    valid = (m > 0) & np.isfinite(disp) & (disp > 0) & (xx - disp >= 0)
    z = np.full((h,w), np.nan, np.float32)
    z[valid] = (K_fs[0,0] * baseline) / disp[valid]
    valid &= (z >= zmin) & (z <= zmax)
    n0 = int(valid.sum())
    if n0 < min_pts:
        return None, total, n0, np.nan, np.nan
    v,u = np.where(valid)
    zz = z[valid].astype(np.float64)
    median_z = np.median(zz)
    mad = np.median(np.abs(zz - median_z))
    tolerance = max(0.03, 3.0 * 1.4826 * mad)
    keep = np.abs(zz-median_z) <= tolerance
    if keep.sum() < min_pts:
        return None, total, n0, float(median_z), float(mad)
    u = u[keep]; v = v[keep]; zz = zz[keep]
    xx3 = (u - K_fs[0,2]) * zz / K_fs[0,0]
    yy3 = (v - K_fs[1,2]) * zz / K_fs[1,1]
    # Undo 90 deg CCW image rotation: FS xyz = (Yrect, -Xrect, Zrect)
    rect_xyz = np.stack((-yy3, xx3, zz), axis=1)
    # R1 maps original Camera1 -> rectified Camera1: X_original = R1.T X_rectified.
    cam1_xyz = rect_xyz @ R1
    med_cam = np.median(cam1_xyz, axis=0)
    med_rect = np.median(rect_xyz, axis=0)
    return {'cam1': med_cam, 'rect': med_rect,
            'u': float(np.median(u)), 'v': float(np.median(v)),
            'n_retained': int(len(zz)), 'depth_median': float(np.median(zz))}, total, n0, float(median_z), float(mad)


def depth(args):
    out = args.out.resolve()
    meta = json.loads((out / 'meta.json').read_text())
    K_fs = np.asarray(meta['K_fs'], np.float64)
    R1 = np.asarray(meta['R1'], np.float64)
    baseline = float(meta['baseline_m'])
    source_indices = meta['source_video_frames']
    if not (out/'masks/00000.png').is_file():
        raise FileNotFoundError('No SAM masks. Run `segment` first.')
    model, InputPadder, torch = load_fs(args)
    from contextlib import nullcontext

    rows = []
    overlays = out / 'xyz_overlays'
    overlays.mkdir(exist_ok=True)
    n = len(source_indices)
    if args.limit > 0:
        n = min(n, args.limit)
    header = ['sequence_frame','source_video_frame','time_from_video_fps_est_s',
              'status','mask_px','valid_depth_px','inlier_depth_px','valid_fraction',
              'X_cam1_m','Y_cam1_m','Z_cam1_m',
              'X_rect_m','Y_rect_m','Z_rect_m',
              'depth_m','median_depth_before_filter_m','depth_mad_m','u_fs_px','v_fs_px']
    for i in range(n):
        file_l = out / 'left_png' / f'{i:05d}.png'
        file_r = out / 'right_png' / f'{i:05d}.png'
        ml = out / 'masks' / f'{i:05d}.png'
        if not ml.is_file():
            raise FileNotFoundError(f'No SAM mask for frame {i}: {ml}')
        bgr1 = cv2.imread(str(file_l)); bgr2 = cv2.imread(str(file_r))
        mask = cv2.imread(str(ml), cv2.IMREAD_GRAYSCALE)
        if bgr1 is None or bgr2 is None or mask is None:
            raise IOError('Missing input image or mask for frame '+str(i))
        if bgr1.shape != bgr2.shape or bgr1.shape[:2] != mask.shape:
            raise ValueError('Size mismatch in prepared left/right or mask at frame '+str(i))
        img1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
        H,W = img1.shape[:2]
        a = torch.from_numpy(np.ascontiguousarray(img1)).cuda().float()[None].permute(0,3,1,2)
        b = torch.from_numpy(np.ascontiguousarray(img2)).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(a.shape, divis_by=32, force_square=False)
        a,b = padder.pad(a,b)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            predicted = model.forward(a,b, iters=args.iters, test_mode=True)
        disp = padder.unpad(predicted.float()).detach().cpu().numpy().reshape(H,W)
        del a,b,predicted
        est,total,n_valid,depth_med,mad = sample_band_xyz(mask, disp, K_fs, R1,baseline,
                                                             args.z_min,args.z_max,
                                                             args.erode,args.min_points)
        fps = float(meta.get('video_fps_reported') or 0)
        # The recorded FPS may not equal the camera acquisition rate.
        video_t = source_indices[i] / fps if fps > 0 else float('nan')
        if est is None:
            status='INVALID_DEPTH'
            cam = [float('nan')]*3
            rect = [float('nan')]*3
            inliers=0; depth_m=float('nan'); u=float('nan'); v=float('nan')
        else:
            status='VALID'
            cam=est['cam1']; rect=est['rect']
            inliers=est['n_retained']; depth_m=est['depth_median']
            u=est['u']; v=est['v']
        rows.append([i,source_indices[i],video_t,status,total,n_valid,inliers,
                     n_valid/max(total,1),*cam,*rect,depth_m,depth_med,mad,u,v])
        if i % args.vis_every == 0:
            vis=bgr1.copy()
            overlay=vis.copy(); overlay[mask>0]=(0,140,0)
            vis=cv2.addWeighted(vis,0.65,overlay,0.35,0)
            if est:
                cv2.circle(vis,(int(round(u)),int(round(v))),6,(0,255,255),-1)
                msg=f'{i:04d} XYZ Cam1: {cam[0]:+.3f} {cam[1]:+.3f} {cam[2]:+.3f}m'
            else:
                msg=f'{i:04d} INVALID, valid depth pts={n_valid}'
            cv2.putText(vis,msg,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.43,(255,255,255),2)
            cv2.imwrite(str(overlays / f'{i:05d}.png'),vis)
        if i % 10 == 0 or i == n-1:
            print(f'Depth {i+1}/{n}: {status}, {inliers} retained depth pixels',flush=True)
    csv_path = out/'wrist_xyz_camera1.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)
    valid_count=sum(r[3]=='VALID' for r in rows)
    print(f'\nDONE: {valid_count}/{len(rows)} frames with wristband 3D')
    print('CSV:',csv_path)
    print('Depth overlays:',overlays)
    print('XYZ refers to VISIBLE wristband surface in original Camera1 optical frame.')
    print('It is NOT yet transformed into table/world coordinates or wrist joint center.')
    print('Time column is estimated from video FPS, not original capture timestamps.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub=parser.add_subparsers(dest='cmd',required=True)
    pp=sub.add_parser('prepare',help='Rectify and rotate synchronized videos; create SAM+FS frames')
    pp.add_argument('--take',type=Path,default=DEFAULT_TAKE)
    pp.add_argument('--calib',type=Path,default=DEFAULT_CALIB)
    pp.add_argument('--out',type=Path,default=DEFAULT_OUT)
    pp.add_argument('--start',type=int,default=0,help='Original video start frame, inclusive')
    pp.add_argument('--end',type=int,default=-1,help='Original video end frame, inclusive (-1=all)')
    pp.add_argument('--step',type=int,default=1)
    pp.add_argument('--max-frames',type=int,default=0,help='0=all selected frames')
    pp.add_argument('--scale',type=float,default=0.5)

    pd=sub.add_parser('depth',help='Extract wristband-surface XYZ with FoundationStereo + SAM3 masks')
    pd.add_argument('--out',type=Path,default=DEFAULT_OUT)
    pd.add_argument('--fs-repo',type=Path,default=DEFAULT_FS)
    pd.add_argument('--fs-ckpt',type=Path,default=DEFAULT_FS/'pretrained_models/11-33-40/model_best_bp2.pth')
    pd.add_argument('--iters',type=int,default=16)
    pd.add_argument('--limit',type=int,default=0,help='0=all prepared frames')
    pd.add_argument('--z-min',type=float,default=0.2)
    pd.add_argument('--z-max',type=float,default=3.0)
    pd.add_argument('--erode',type=int,default=3)
    pd.add_argument('--min-points',type=int,default=20)
    pd.add_argument('--vis-every',type=int,default=10)
    args=parser.parse_args()
    if args.cmd=='prepare': prepare(args)
    elif args.cmd=='depth': depth(args)


if __name__=='__main__':
    main()
