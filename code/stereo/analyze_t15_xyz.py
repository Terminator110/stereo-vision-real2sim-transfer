#!/usr/bin/env python3
"""T15 XYZ continuity, stationary jitter and optional known-displacement analysis.

The XYZ standard deviations measure repeatability only when the wrist is stationary.
They do NOT establish absolute anatomical accuracy without independent ground truth.
"""
import argparse
import numpy as np
import pandas as pd


def stats(values):
    a = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float).ravel()
    a = a[np.isfinite(a)]
    if not len(a): return 'n/a'
    return f'median={np.median(a):.4f} p90={np.percentile(a,90):.4f} max={a.max():.4f}'


def runs(mask):
    best=cur=0
    for v in mask:
        cur=cur+1 if v else 0
        best=max(best,cur)
    return best


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='tests/T15_wrist_xyz.csv')
    p.add_argument('--interval',type=float,default=5.0)
    p.add_argument('--duration',type=float,default=None)
    p.add_argument('--scale-to-mm',type=float,default=1000.0,
                   help='Multiply XYZ calibration units by this value for mm. Default assumes stereo T in metres. Use 1 if T in mm.')
    args=p.parse_args()
    df=pd.read_csv(args.csv)
    df['t']=df.time_s-df.time_s.iloc[0]
    good=df['mode'].eq('STEREO_WRIST')
    both=df.left_detected.eq(1)&df.right_detected.eq(1)
    dt=np.median(np.diff(df.time_s)) if len(df)>1 else np.nan
    print('========= T15 WRIST XYZ DIAGNOSTICS =========')
    print('Total frames:',len(df),' | Recording duration:',round(float(df.t.iloc[-1]),2),'s')
    print('Median processed frame rate:',round(float(1/dt),2) if dt>0 else 'n/a','Hz')
    print('Valid wrist XYZ:',int(good.sum()),f'({good.mean()*100:.1f}%)')
    print('Both cameras:',int(both.sum()),f'({both.mean()*100:.1f}%)')
    print('Longest consecutive INVALID frames:',runs(~good.to_numpy()))
    print('\n=== REJECTION REASONS ===')
    print(df.reason.value_counts(dropna=False).to_string())
    print('\n=== ERROR DISTRIBUTION (INCLUDING REJECTED DUAL DETECTIONS) ===')
    print('Wrist reproj, px:',stats(df.loc[both,'wrist_reproj_px']))
    print('MCP reproj, px:',stats(df.loc[both,'mcp_reproj_median_px']))
    print('Valid MCP count (dual detection):',stats(df.loc[both,'valid_mcp_count']))
    print('\n=== TIME INTERVALS ===')
    stop=float(df.t.iloc[-1])+1e-6 if args.duration is None else min(args.duration,float(df.t.iloc[-1])+1e-6)
    for a in np.arange(0,stop,args.interval):
        b=a+args.interval
        s=df[(df.t>=a)&(df.t<b)]
        if s.empty: continue
        valid=s['mode'].eq('STEREO_WRIST'); dual=s.left_detected.eq(1)&s.right_detected.eq(1)
        print(f'\n{a:g}-{b:g} s | frames={len(s)} both={int(dual.sum())} valid={int(valid.sum())} ({valid.mean()*100:.1f}%)')
        print('  Wrist error:',stats(s.loc[dual,'wrist_reproj_px']))
        for col in ('x','y','z'):
            values=pd.to_numeric(s.loc[valid,col],errors='coerce').dropna()
            if len(values)>=3:
                values=values.to_numpy()*args.scale_to_mm
                median=np.median(values)
                mad=np.median(abs(values-median))
                print(f'  {col.upper()}: median={median:+.1f} mm  std={np.std(values,ddof=1):.1f} mm  MAD={mad:.1f} mm  p95-p5={np.percentile(values,95)-np.percentile(values,5):.1f} mm')
            else:
                print(f'  {col.upper()}: insufficient accepted samples')
        print('  NOTE: axis variation is *jitter* only if the wrist stayed stationary.')
    print('\n=== CONSECUTIVE ACCEPTED XYZ STEPS ===')
    coords=df[['x','y','z']].apply(pd.to_numeric,errors='coerce').to_numpy()
    t=df.time_s.to_numpy(dtype=float)
    mask=good.to_numpy()
    step=[]; speed=[]
    for i in range(1,len(df)):
        if mask[i] and mask[i-1] and t[i]>t[i-1]:
            length=float(np.linalg.norm(coords[i]-coords[i-1])*args.scale_to_mm)
            step.append(length);speed.append(length/(t[i]-t[i-1]))
    print('Step length (mm):', stats(step))
    print('Implied speed (mm/s):',stats(speed))
    print('Step/speed include genuine movement: inspect stationary segments separately.')
    print('\n=== PALM PROXY AVAILABILITY (NOT WRIST) ===')
    palm=df[['palm_x','palm_y','palm_z']].notna().all(axis=1)
    print('Palm MCP-centre usable:',int(palm.sum()),f'({palm.mean()*100:.1f}%)')
    print('Palm usable while wrist rejected:',int((palm&~good).sum()))
    print('\nIMPORTANT: absolute XYZ error requires an external positional reference.')

if __name__=='__main__': main()
