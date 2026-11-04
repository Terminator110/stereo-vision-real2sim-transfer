import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

take = Path(sys.argv[1])
annotation_file = Path("annotation1.json")
output = take / "mujoco_export"
output.mkdir(exist_ok=True)

# Read Label Studio annotation.
tasks = json.loads(annotation_file.read_text())

ranges = []

for task in tasks:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            value = result.get("value", {})

            if "Approach" in value.get("timelinelabels", []):
                for interval in value.get("ranges", []):
                    ranges.append((
                        int(interval["start"]),
                        int(interval["end"])
                    ))

if len(ranges) != 1:
    raise RuntimeError(
        f"Expected one Approach interval, found {ranges}"
    )

start, end = ranges[0]
print(f"Approach: frames {start} to {end} (end exclusive)")

# Use acquisition timestamps, not assumed 20 FPS timing.
timestamps = pd.read_csv(take / "timestamps.csv")
timestamps = timestamps.drop_duplicates("frame").set_index("frame")

if start not in timestamps.index:
    raise RuntimeError("Approach start frame missing from timestamps")

start_time = float(timestamps.loc[start, "time_s"])

# Use calibrated TABLE coordinates.
source = take / "tags_xyz_table.csv"

if not source.exists():
    raise RuntimeError(
        f"Missing {source}. Convert Camera 1 XYZ to table XYZ first."
    )

df = pd.read_csv(source)

required = [
    "frame", "tag_id", "valid",
    "x_table_m", "y_table_m", "z_table_m"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing CSV columns: {missing}")

df = df[(df.frame >= start) & (df.frame < end)].copy()

total_frames = end - start

for tag_id in (2, 4, 5):
    tag = df[df.tag_id == tag_id].copy()

    xyz_cols = ["x_table_m", "y_table_m", "z_table_m"]

    good = (
        tag["valid"].eq(1)
        & np.isfinite(tag[xyz_cols].to_numpy(dtype=float)).all(axis=1)
    )

    tag = tag.loc[good].sort_values("frame")

    if len(tag):
        missing_timestamps = [
            int(f) for f in tag.frame
            if f not in timestamps.index
        ]
        if missing_timestamps:
            raise RuntimeError(
                f"Missing frame timestamps: {missing_timestamps}"
            )

        times = timestamps.loc[tag.frame, "time_s"].to_numpy(dtype=float)
        times = times - start_time

        result = pd.DataFrame({
            "time_s": times,
            "side": "right",
            "x": tag["x_table_m"].to_numpy(),
            "y": tag["y_table_m"].to_numpy(),
            "z": tag["z_table_m"].to_numpy(),
        })

        # Gap statistics include missing frames at both clip boundaries.
        measured_frames = tag.frame.to_numpy(dtype=int)
        boundaries = np.r_[start - 1, measured_frames, end]
        missing_runs = np.diff(boundaries) - 1
        longest_gap = int(missing_runs.max())

    else:
        result = pd.DataFrame(columns=["time_s", "side", "x", "y", "z"])
        longest_gap = total_frames

    path = output / f"approach_tag{tag_id}_table.csv"
    result.to_csv(path, index=False)

    coverage = 100 * len(result) / total_frames

    print(
        f"Tag {tag_id}: {len(result)}/{total_frames} frames "
        f"({coverage:.1f}%), longest missing run "
        f"{longest_gap} frames → {path}"
    )

metadata = {
    "action": "Approach",
    "start_frame": start,
    "end_frame_exclusive": end,
    "coordinate_frame": "calibrated_table_v3",
    "units": "metres",
    "point_type": "AprilTag centre",
    "orientation": "not available",
    "interpolated": False,
    "mujoco_alignment_applied": False,
    "hand_base_offset_applied": False,
}

(output / "metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

print("\nExport complete:", output)
