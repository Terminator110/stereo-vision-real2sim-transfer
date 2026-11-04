import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("csv", help="Recorded wrist trajectory CSV")
p.add_argument("--expected-mm", type=float, default=None,
               help="Known physical displacement between A and B")
p.add_argument("--a", nargs=2, type=float, metavar=("START", "END"),
               help="Stationary interval A, seconds from recording start")
p.add_argument("--b", nargs=2, type=float, metavar=("START", "END"),
               help="Stationary interval B, seconds from recording start")
p.add_argument("--plot", action="store_true")
args = p.parse_args()

df = pd.read_csv(args.csv)
if df.empty:
    raise SystemExit("CSV is empty")

required = [
    "time_s", "valid", "x_cam1_m", "y_cam1_m", "z_cam1_m",
    "mean_reprojection_px", "mean_tag_edge_mm"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

df = df.sort_values("time_s").reset_index(drop=True)
df["t"] = df.time_s - df.time_s.iloc[0]

xyzcols = ["x_cam1_m", "y_cam1_m", "z_cam1_m"]
valid = (df.valid == 1) & np.isfinite(df[xyzcols]).all(axis=1)

d = df.loc[valid].copy()

n = len(df)
nv = len(d)
duration = float(df.t.iloc[-1]) if n > 1 else 0.0

print("\n========== STEREO WRIST VALIDATION ==========")
print("File:", Path(args.csv).name)
print(f"Frames: {n}")
print(f"Duration: {duration:.2f} s")
print(f"Valid: {nv}/{n} ({100 * nv/n:.2f}%)")

if n > 1:
    dt_all = np.diff(df.time_s.to_numpy(float))
    dt_positive = dt_all[dt_all > 0]
    if len(dt_positive):
        frame_dt = np.median(dt_positive)
        print(f"Recorded sampling rate: {1/frame_dt:.2f} Hz (median)")
    else:
        frame_dt = 0.05
else:
    frame_dt = 0.05

# Consecutive invalid samples.
bad = (~valid).to_numpy()
runs = []
i = 0
while i < n:
    if not bad[i]:
        i += 1
        continue
    j = i + 1
    while j < n and bad[j]:
        j += 1
    # Duration estimated from observation timestamps.
    gap_s = float((df.time_s.iloc[j] - df.time_s.iloc[i])
                  if j < n else
                  (df.time_s.iloc[j-1] - df.time_s.iloc[i] + frame_dt))
    runs.append((i, j-i, gap_s))
    i = j

print(f"Invalid runs: {len(runs)}")
if runs:
    worst = max(runs, key=lambda x: x[2])
    print(f"Longest invalid run: {worst[1]} samples, "
          f"approximately {worst[2]:.3f} s")

if nv < 2:
    raise SystemExit("Not enough valid points")

xyz = d[xyzcols].to_numpy(float) * 1000.0
ts = d.t.to_numpy(float)

print("\n--- Calibration consistency ---")
err = d.mean_reprojection_px.to_numpy(float)
edge = d.mean_tag_edge_mm.to_numpy(float)
print(f"Reprojection mean: {np.nanmean(err):.3f} px")
print(f"Reprojection 95th percentile: "
      f"{np.nanpercentile(err,95):.3f} px")
print(f"Tag edge mean: {np.nanmean(edge):.3f} mm")
print(f"Tag edge std: {np.nanstd(edge,ddof=1):.3f} mm")

print("\n--- XYZ movement extent (camera coordinates) ---")
print("X/Y/Z range mm:", np.ptp(xyz, axis=0).round(2))
print("3D XYZ standard deviations mm:",
      np.std(xyz,axis=0,ddof=1).round(2))

# Speed calculation: only adjacent original frames, not across dropouts.
idx = d.index.to_numpy()
adjacent = np.diff(idx) == 1
td = np.diff(ts)
step = np.linalg.norm(np.diff(xyz,axis=0),axis=1)
good_speed = adjacent & (td > 0)
speed = step[good_speed] / td[good_speed] if np.any(good_speed) else np.array([])
if len(speed):
    print(f"Median observed speed: {np.median(speed):.1f} mm/s")
    print(f"95th percentile speed: {np.percentile(speed,95):.1f} mm/s")
    print("Largest consecutive-frame displacement:",
          round(float(np.max(step[good_speed])),2), "mm")

print("Cumulative valid adjacent-frame path:",
      round(float(np.sum(step[adjacent])),2), "mm")

# Endpoint windows. Defaults: first/last 3 seconds.
def window(interval, default_start, default_end):
    a, b = interval if interval is not None else (default_start,default_end)
    subset = d[(d.t >= a) & (d.t <= b)]
    return subset, (a,b)

a,aw = window(args.a,0,min(3.0,duration))
b,bw = window(args.b,max(0,duration-3.0),duration)

def stats(s,label,w):
    if len(s)<5:
        print(f"{label}: too few valid frames in interval {w}")
        return None
    v=s[xyzcols].to_numpy(float)*1000
    mean=np.mean(v,axis=0)
    sd=np.std(v,axis=0,ddof=1)
    print(f"{label} ({w[0]:.2f}-{w[1]:.2f}s, n={len(s)}):")
    print("  Mean XYZ mm:",mean.round(2))
    print("  STD XYZ mm: ",sd.round(2))
    return mean

print("\n--- Stationary-window comparison ---")
ma=stats(a,"A",aw)
mb=stats(b,"B",bw)

if ma is not None and mb is not None:
    delta=mb-ma
    dist=float(np.linalg.norm(delta))
    print("B - A XYZ displacement (mm):",delta.round(2))
    print(f"A to B 3D distance: {dist:.2f} mm")
    if args.expected_mm is not None:
        e=dist-args.expected_mm
        print(f"Expected displacement: {args.expected_mm:.2f} mm")
        print(f"Absolute distance error: {abs(e):.2f} mm")
        print(f"Signed error: {e:+.2f} mm")
        print(f"Relative absolute error: "
              f"{100*abs(e)/args.expected_mm:.2f}%"
              if args.expected_mm>0 else "Expected must be >0")

out=Path(args.csv).with_suffix("")
if args.plot:
    fig, axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
    labels=["X","Y","Z"]
    for i,ax in enumerate(axs):
        ax.plot(ts,xyz[:,i],label=labels[i])
        ax.axvspan(*aw,alpha=0.1)
        ax.axvspan(*bw,alpha=0.1)
        ax.set_ylabel(f"{labels[i]} (mm)")
        ax.grid(True)
        ax.legend()
    axs[-1].set_xlabel("Time since recording start (s)")
    fig.suptitle("Wrist trajectory | shaded: A/B measurement windows")
    fig.tight_layout()
    f1=Path(str(out)+"_xyz.png")
    fig.savefig(f1,dpi=150)
    plt.close(fig)

    fig=plt.figure(figsize=(9,7))
    ax=fig.add_subplot(111,projection="3d")
    ax.plot(xyz[:,0],xyz[:,1],xyz[:,2],lw=1.5)
    ax.scatter(*xyz[0],marker="o",s=50,label="Start")
    ax.scatter(*xyz[-1],marker="x",s=60,label="End")
    ax.set_xlabel("Camera X (mm)")
    ax.set_ylabel("Camera Y (mm)")
    ax.set_zlabel("Camera Z (mm)")
    ax.legend()
    fig.tight_layout()
    f2=Path(str(out)+"_3d.png")
    fig.savefig(f2,dpi=150)
    plt.close(fig)
    print("\nPlots saved:",f1,f2)

print("\nNote: Coordinates are in Camera 1 frame, NOT table frame.")
print("Endpoint accuracy is meaningful only for known physical motion")
print("and genuinely stationary A/B windows.")
