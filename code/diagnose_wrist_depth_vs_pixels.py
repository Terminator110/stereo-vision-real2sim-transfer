from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TAKE = "take_20261104_163208_670689"

TRIAL = (
    Path.home()
    / "physikal_workflow/dex-5/FoundationStereo"
    / f"wristband_batch_{TAKE}"
)

df = pd.read_csv(TRIAL / "wrist_xyz_camera1.csv")

frames = df["source_video_frame"].to_numpy()
u = df["u_fs_px"].to_numpy(float)
v = df["v_fs_px"].to_numpy(float)
depth = df["Z_cam1_m"].to_numpy(float)
fraction = df["valid_fraction"].to_numpy(float)

fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

axes[0].plot(frames, u)
axes[0].set_ylabel("Wristband U (px)")

axes[1].plot(frames, v)
axes[1].set_ylabel("Wristband V (px)")

axes[2].plot(frames, depth)
axes[2].set_ylabel("Camera Z (m)")

axes[3].plot(frames, fraction)
axes[3].set_ylabel("Valid depth fraction")
axes[3].set_xlabel("Source video frame")

for ax in axes:
    ax.axvspan(112, 116, alpha=0.12)
    ax.axvspan(175, 197, alpha=0.08)
    ax.grid(True)

fig.suptitle("Wristband 2D Tracking vs FoundationStereo Depth")
fig.tight_layout()

output = Path("/tmp/dex5_wrist_depth_vs_pixels.png")
fig.savefig(output, dpi=160)

print("Saved:", output)

for a, b in [(108, 119), (175, 197)]:
    print(f"\n========== FRAMES {a}-{b} ==========")
    print(
        df.loc[
            df.source_video_frame.between(a, b),
            [
                "source_video_frame",
                "u_fs_px",
                "v_fs_px",
                "Z_cam1_m",
                "valid_fraction",
            ]
        ].round(3).to_string(index=False)
    )
