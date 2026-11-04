import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).resolve().parent
df = pd.read_csv(base / "wrist_trajectory.csv")

t = df["time_s"].to_numpy()
valid = df["valid"] == 1

fig, ax = plt.subplots(figsize=(12, 6))

for col, name in [
    ("x_cam1_m", "X"),
    ("y_cam1_m", "Y"),
    ("z_cam1_m", "Z"),
]:
    ax.plot(
        t[valid],
        df.loc[valid, col] * 1000,
        label=name
    )

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Position relative to Camera 1 (mm)")
ax.set_title("Stereo Wrist Trajectory")
ax.legend()
ax.grid(True)

fig.tight_layout()
out = base / "wrist_trajectory.png"
fig.savefig(out, dpi=160)

print("Saved:", out)

plt.show()
