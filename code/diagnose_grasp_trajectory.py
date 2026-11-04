from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "physikal_workflow/dex-5"
TAKE = "take_20261104_163208_670689"

for suffix in ("", "_swapped"):
    label = "ORIGINAL" if not suffix else "SWAPPED"

    sim_path = (
        ROOT / "outputs"
        / f"smoothed_{TAKE}{suffix}"
        / "sim_quality.csv"
    )

    palm_path = (
        ROOT / "FoundationStereo"
        / f"wristband_batch_{TAKE}{suffix}"
        / "palm_pose_6dof.csv"
    )

    if not sim_path.exists() or not palm_path.exists():
        print(f"{label}: Missing CSV; skipping")
        continue

    sim = pd.read_csv(sim_path)
    palm = pd.read_csv(palm_path)

    target = sim[
        ["wrist_target_x", "wrist_target_y", "wrist_target_z"]
    ].to_numpy(dtype=float)

    actual = sim[
        ["wrist_actual_x", "wrist_actual_y", "wrist_actual_z"]
    ].to_numpy(dtype=float)

    table = palm[
        ["X_table_m", "Y_table_m", "Z_table_m"]
    ].to_numpy(dtype=float)

    print(f"\n========== {label} ==========")

    for name, xyz in [
        ("RECONSTRUCTED TABLE", table),
        ("MUJOCO TARGET", target),
        ("MUJOCO ACTUAL", actual)
    ]:
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        steps[~np.isfinite(steps)] = np.nan

        print(f"\n{name}:")
        valid = np.flatnonzero(np.isfinite(steps))

        for i in sorted(
            valid,
            key=lambda j: steps[j],
            reverse=True
        )[:8]:
            print(
                f"  Frame {i}->{i+1}: "
                f"{steps[i]*1000:.1f} mm"
            )

    print("\n========== FINAL 20 FRAMES ==========")

    cols = [
        "source_video_frame",
        "ik_status",
        "pose_source",
        "position_error_mm",
        "orientation_error_deg",
        "wrist_target_x",
        "wrist_target_y",
        "wrist_target_z",
        "wrist_actual_x",
        "wrist_actual_y",
        "wrist_actual_z",
    ]

    print(
        sim[cols].tail(20).round(3).to_string(index=False)
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    frames = sim["source_video_frame"].to_numpy()

    for k, axis in enumerate("XYZ"):
        axes[k].plot(
            frames, target[:, k],
            label=f"Target {axis}"
        )
        axes[k].plot(
            frames, actual[:, k],
            label=f"Actual {axis}"
        )

        axes[k].set_ylabel(f"{axis} (m)")
        axes[k].grid(True)
        axes[k].legend()

    axes[-1].set_xlabel("Source video frame")
    fig.suptitle(f"{label}: MuJoCo Wrist Target vs Actual")
    fig.tight_layout()

    output = f"/tmp/dex5_{label.lower()}_trajectory.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)

    print("\nSaved plot:", output)
