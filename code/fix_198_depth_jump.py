from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "physikal_workflow/dex-5"
TAKE = "take_20261104_163208_670689"

TRIAL = ROOT / "FoundationStereo" / f"wristband_batch_{TAKE}"

src = TRIAL / "palm_pose_6dof.csv"
dst = TRIAL / "palm_pose_6dof_depth_fixed.csv"

df = pd.read_csv(src)

cols = ["X_table_m", "Y_table_m", "Z_table_m"]

frames = df["source_video_frame"].to_numpy(int)

start = 112
end = 116

a = df.loc[frames == start, cols].to_numpy(float)
b = df.loc[frames == end, cols].to_numpy(float)

assert a.shape == (1, 3)
assert b.shape == (1, 3)
assert np.isfinite(a).all() and np.isfinite(b).all()

for frame in range(start + 1, end):

    alpha = (frame - start) / (end - start)

    interpolated = (
        (1 - alpha) * a[0] +
        alpha * b[0]
    )

    df.loc[
        df["source_video_frame"] == frame,
        cols
    ] = interpolated

df.to_csv(dst, index=False)

print("Corrected:", dst)
print("Frames modified: 113, 114, 115")
print("Original preserved.")

print("\nCorrected XYZ:")
print(
    df.loc[
        df.source_video_frame.between(110, 118),
        ["source_video_frame"] + cols
    ].round(4).to_string(index=False)
)
