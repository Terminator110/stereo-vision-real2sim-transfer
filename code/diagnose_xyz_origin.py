from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "physikal_workflow/dex-5/FoundationStereo"
TAKE = "take_20261104_163208_670689"

for suffix in ("", "_swapped"):
    name = "ORIGINAL" if not suffix else "SWAPPED"
    folder = ROOT / f"wristband_batch_{TAKE}{suffix}"

    raw = pd.read_csv(folder / "wrist_xyz_camera1.csv")
    palm = pd.read_csv(folder / "palm_pose_6dof.csv")

    df = raw.merge(
        palm,
        on="source_video_frame",
        suffixes=("_depth", "_palm"),
        validate="one_to_one"
    ).sort_values("source_video_frame").reset_index(drop=True)

    camera = df[
        ["X_cam1_m", "Y_cam1_m", "Z_cam1_m"]
    ].to_numpy(float)

    table = df[
        ["X_table_m", "Y_table_m", "Z_table_m"]
    ].to_numpy(float)

    cam_step = np.linalg.norm(np.diff(camera, axis=0), axis=1) * 1000
    table_step = np.linalg.norm(np.diff(table, axis=0), axis=1) * 1000

    print(f"\n========== {name} ==========")

    for start, end in [(105, 121), (175, 197)]:
        print(f"\nFRAMES {start}–{end}")
        print(
            "frame | depth Z | camera step | table step | "
            "mask pixels | depth fraction | status"
        )

        for i in range(start, min(end + 1, len(df))):
            r = df.iloc[i]
            cs = cam_step[i-1] if i > 0 else np.nan
            ts = table_step[i-1] if i > 0 else np.nan

            print(
                f"{int(r.source_video_frame):5d} | "
                f"{r.Z_cam1_m:7.3f} | "
                f"{cs:11.1f} | "
                f"{ts:10.1f} | "
                f"{int(r.mask_px):11d} | "
                f"{r.valid_fraction:14.3f} | "
                f"{r.status}"
            )

    print("\n========== CAMERA VS TABLE JUMP AGREEMENT ==========")

    both_valid = (
        np.isfinite(camera).all(axis=1)
        & np.isfinite(table).all(axis=1)
    )

    valid_pairs = both_valid[:-1] & both_valid[1:]

    diff = np.abs(cam_step - table_step)

    if valid_pairs.any():
        print(
            "Median difference in step lengths (mm):",
            round(float(np.median(diff[valid_pairs])), 4)
        )

        print(
            "Maximum difference in step lengths (mm):",
            round(float(np.max(diff[valid_pairs])), 4)
        )

        if np.max(diff[valid_pairs]) < 1:
            print(
                "RESULT: Table transformation preserves movement lengths."
            )
            print(
                "Large position jumps already exist in camera-space XYZ."
            )
        else:
            print(
                "RESULT: Movement lengths change between camera and table."
            )
            print(
                "Inspect camera-to-table conversion or palm-pose combination."
            )

