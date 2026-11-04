#!/usr/bin/env python3
"""SAM3 object keyframe localization for upright cup and flat coaster.

Requires offline processor's object_keyframes folder. CUP IS ASSUMED ON TABLE.
Outputs camera masks/diagnostics and static-placement estimates, NOT lifted cup 6DoF.
"""
import argparse
import csv
import re
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import minimize
from ultralytics.models.sam import SAM3SemanticPredictor

def centroid(poly):
    x,y=poly[:,0],poly[:,1]
    cross=x*np.roll(y,-1)-np.roll(x,-1)*y
    area=np.sum(cross)
    if abs(area)<1e-10:return None
    return np.array([np.sum((x+np.roll(x,-1))*cross)/(3*area),np.sum((y+np.roll(y,-1))*cross)/(3*area)])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--take',type=Path,required=True)
    ap.add_argument('--calib',type=Path,default=Path('stereo/calibration/results_new/stereo_calibration.npz'))
    ap.add_argument('--table',type=Path,default=Path('stereo/calibration/results_new/table_frame_multi.npz'))
    ap.add_argument('--model',default='/home/sparx3d/physikal_workflow/sam3.pt')
    ap.add_argument('--cup-radius-mm',type=float,default=41)
    ap.add_argument('--cup-height-mm',type=float,default=98)
    ap.add_argument('--coaster-top-mm',type=float,default=0.,help='Coaster visible surface height above tabletop; 0 for very thin')
    ap.add_argument('--confidence',type=float,default=.25)
    ap.add_argument('--min-cup-dice',type=float,default=.75)
    ap.add_argument('--max-coaster-disagreement-mm',type=float,default=25.)
    args=ap.parse_args()
    src=args.take/'processed'/'object_keyframes'
    if not src.exists():raise SystemExit('Missing object keyframes. Re-run process_recorded_stereo.py --object-every-s 1')
    out=args.take/'processed'/'object_localization';out.mkdir(parents=True,exist_ok=True)
    with np.load(args.calib) as f: c={k:np.asarray(f[k],np.float64) for k in ('K1','D1','K2','D2','R','T')}
    with np.load(args.table) as f:
        R1=np.asarray(f['R_table_to_cam1'],np.float64)
        t1=np.asarray(f['t_table_to_cam1'],np.float64).reshape(3,1)
    RR=[R1,c['R']@R1];tt=[t1,c['R']@t1+c['T'].reshape(3,1)]
    KK=[c['K1'],c['K2']];DD=[c['D1'],c['D2']]
    centers=[(-rot.T@t).reshape(3) for rot,t in zip(RR,tt)]
    rv=[cv2.Rodrigues(r)[0] for r in RR]
    rad=args.cup_radius_mm/1000; height=args.cup_height_mm/1000
    angles=np.linspace(0,2*np.pi,96,endpoint=False)
    cylinder=np.array([[rad*np.cos(a),rad*np.sin(a),z] for z in (0,height) for a in angles],np.float64)
    def uv(points,cam):
        q,_=cv2.projectPoints(points,rv[cam],tt[cam],KK[cam],DD[cam]);return q.reshape(-1,2)
    def plane_pixels(pixels,cam,z):
        q=cv2.undistortPoints(pixels.reshape(-1,1,2),KK[cam],DD[cam]).reshape(-1,2)
        rays=(RR[cam].T@np.column_stack([q,np.ones(len(q))]).T).T
        lam=(z-centers[cam][2])/rays[:,2]
        good=np.isfinite(lam)&(lam>0)&(np.abs(rays[:,2])>1e-8)
        if not good.all():return None
        return centers[cam]+lam[:,None]*rays
    def cylinder_render(xy,cam,shape):
        pts=cylinder+np.array([xy[0],xy[1],0])
        if np.any((RR[cam]@pts.T+tt[cam])[2]<=0):return None
        poly=cv2.convexHull(np.round(uv(pts,cam)).astype(np.int32))
        m=np.zeros(shape,np.uint8);cv2.fillConvexPoly(m,poly,1);return m
    def dice(a,b):
        denom=a.sum()+b.sum()
        return 2*np.count_nonzero(a&b)/denom if denom else 0
    def get_mask(predictor,path,text):
        predictor.set_image(str(path))
        results=predictor(text=[text]); predictor.reset_image()
        if not results or results[0].masks is None:return None
        masks=results[0].masks.data.cpu().numpy()
        if not len(masks):return None
        m=(masks[np.argmax(masks.reshape(len(masks),-1).sum(axis=1))]>.5).astype('uint8')
        image=cv2.imread(str(path));h,w=image.shape[:2]
        if m.shape!=(h,w):m=cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
        return m
    def mask_centre_approx(m,cam,z):
        ys,xs=np.nonzero(m)
        if not len(xs):return None
        pts=np.array([[float(np.mean(xs)),float(np.mean(ys))]],np.float64)
        projected=plane_pixels(pts,cam,z)
        return projected[0,:2] if projected is not None else None
    predictor=SAM3SemanticPredictor(overrides={'model':args.model,'task':'segment','mode':'predict',
                                               'conf':args.confidence,'verbose':False})
    pairs=sorted(src.glob('frame_*_camera1.png'))
    fields=['frame','cup_valid_tabletop','cup_x_table_m','cup_y_table_m','cup_base_z_table_m',
            'cup_centre_z_table_m','cup_dice_cam1','cup_dice_cam2','coaster_valid',
            'coaster_x_table_m','coaster_y_table_m','coaster_top_z_table_m',
            'coaster_disagreement_mm','notes']
    rows=[]
    for pair in pairs:
        match=re.fullmatch(r'frame_(\d+)_camera1.png',pair.name)
        if not match:continue
        index=int(match.group(1));pair2=src/f'frame_{index:06d}_camera2.png'
        if not pair2.exists():continue
        paths=[pair,pair2];frames=[cv2.imread(str(p)) for p in paths]
        row=dict.fromkeys(fields,'');row.update(frame=index,cup_valid_tabletop=0,coaster_valid=0)
        masks={}
        for name,prompt in [('cup','cup'),('coaster','coaster')]:
            masks[name]=[]
            for cam in (0,1):
                m=get_mask(predictor,paths[cam],prompt)
                masks[name].append(m)
                if m is not None:
                    cv2.imwrite(str(out/f'{index:06d}_{name}_cam{cam+1}_mask.png'),m*255)
        # Cup: optimize XY constrained to tabletop. Handle remains unmodeled.
        if all(m is not None and m.sum()>20 for m in masks['cup']):
            init=[mask_centre_approx(masks['cup'][cam],cam,height/2) for cam in (0,1)]
            init=[x for x in init if x is not None]
            if init:
                start=np.mean(init,axis=0)
                def objective(x):
                    d=[]
                    for cam in (0,1):
                        rendered=cylinder_render(x,cam,masks['cup'][cam].shape)
                        if rendered is None:return 2.
                        d.append(1-dice(rendered,masks['cup'][cam]))
                    return np.mean(d)
                bounds=[(start[0]-.30,start[0]+.30),(start[1]-.30,start[1]+.30)]
                res=min([minimize(objective,s,method='Powell',bounds=bounds,
                                  options={'maxiter':100,'xtol':1e-4,'ftol':1e-4}) for s in [start,*init]],key=lambda r:r.fun)
                dd=[dice(cylinder_render(res.x,cam,masks['cup'][cam].shape),masks['cup'][cam]) for cam in (0,1)]
                row.update(cup_x_table_m=f'{res.x[0]:.6f}',cup_y_table_m=f'{res.x[1]:.6f}',
                           cup_base_z_table_m='0.000000',cup_centre_z_table_m=f'{height/2:.6f}',
                           cup_dice_cam1=f'{dd[0]:.4f}',cup_dice_cam2=f'{dd[1]:.4f}',
                           cup_valid_tabletop=int(min(dd)>=args.min_cup_dice))
                for cam in (0,1):
                    overlay=frames[cam].copy()
                    hull=cv2.convexHull(np.round(uv(cylinder+np.array([*res.x,0]),cam)).astype(np.int32))
                    cv2.polylines(overlay,[hull],True,(0,255,0),2)
                    cv2.imwrite(str(out/f'{index:06d}_cup_fit_cam{cam+1}.png'),overlay)
        # Coaster: silhouette pixels assumed on the horizontal top plane.
        if all(m is not None and m.sum()>20 for m in masks['coaster']):
            pos=[]
            for cam in (0,1):
                cont,_=cv2.findContours(masks['coaster'][cam],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if not cont:break
                contour=max(cont,key=cv2.contourArea).reshape(-1,2).astype(np.float64)
                pp=plane_pixels(contour,cam,args.coaster_top_mm/1000)
                if pp is None:break
                centre=centroid(pp[:,:2]);
                if centre is None:break
                pos.append(centre)
            if len(pos)==2:
                disagreement=float(np.linalg.norm(pos[0]-pos[1])*1000)
                mean=np.mean(pos,axis=0)
                row.update(coaster_x_table_m=f'{mean[0]:.6f}',coaster_y_table_m=f'{mean[1]:.6f}',
                           coaster_top_z_table_m=f'{args.coaster_top_mm/1000:.6f}',
                           coaster_disagreement_mm=f'{disagreement:.2f}',
                           coaster_valid=int(disagreement<=args.max_coaster_disagreement_mm))
        row['notes']='cup assumes upright/on table; coaster visible and flat; SAM may fail under occlusion'
        rows.append(row)
        print(f'frame={index} cup_valid={row["cup_valid_tabletop"]} coaster_valid={row["coaster_valid"]} '
              f'cup_dice={row["cup_dice_cam1"]}/{row["cup_dice_cam2"]} coaster_diff={row["coaster_disagreement_mm"]}')
    with (out/'objects_keyframes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print('Saved',out/'objects_keyframes.csv')
    print('WARNING: no lifted-cup pose, handle yaw, coaster yaw, or full object tracking is computed.')

if __name__=='__main__':main()
