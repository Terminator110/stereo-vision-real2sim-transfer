#!/usr/bin/env python3

from pathlib import Path
import csv
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "physikal_workflow/dex-5"
FS = ROOT / "FoundationStereo"
OUT = ROOT / "outputs"
TAKE = "take_20261104_163208_670689"

DEST = OUT / "diagnostics" / TAKE
DEST.mkdir(parents=True, exist_ok=True)

REPORT = []

def log(*args):
    msg = " ".join(map(str, args))
    print(msg)
    REPORT.append(msg)

def normalize(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def find_xyz_groups(columns):
    """
    Find explicitly named X/Y/Z triplets.
    Avoid treating unrelated numeric columns as coordinates.
    """
    groups = {}
    for col in columns:
        name = str(col)
        match = re.match(r"^(.*?)(?:_)?([XYZxyz])(?:_m)?$", name)
        if not match:
            continue
        prefix = normalize(match.group(1))
        axis = match.group(2).lower()
        groups.setdefault(prefix, {})[axis] = col

    return {
        key: axes
        for key, axes in groups.items()
        if set(axes) == {"x", "y", "z"}
    }

def plot_xyz(df, file, label):
    groups = find_xyz_groups(df.columns)

    if not groups:
        log("  XYZ columns: not automatically identifiable")
        return

    for group, axes in groups.items():
        xyz = df[
            [axes["x"], axes["y"], axes["z"]]
        ].apply(pd.to_numeric, errors="coerce").to_numpy()

        valid = np.isfinite(xyz).all(axis=1)
        if valid.sum() < 3:
            continue

        # Show raw units, without guessing coordinate frames.
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # NaN values break lines across missing frames.
        xyz[~valid] = np.nan

        ax.plot(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            linewidth=1.5
        )

        first = np.flatnonzero(valid)[0]
        ax.scatter(
            *xyz[first],
            s=50,
            label=f"Start frame {first}"
        )

        ax.set_xlabel(axes["x"])
        ax.set_ylabel(axes["y"])
        ax.set_zlabel(axes["z"])
        ax.set_title(
            f"{label}\n{file.name}\n"
            f"Coordinate group: {group or 'XYZ'}"
        )
        ax.legend()

        fig.tight_layout()

        name = (
            label.replace(" ", "_")
            + "_"
            + (group or "xyz")
            + ".png"
        )

        fig.savefig(DEST / name, dpi=160)
        plt.close(fig)
        log("  PLOT:", DEST / name)

def inspect_csv(file, label):
    log("\n==========", label, "==========")
    log("FILE:", file)

    try:
        df = pd.read_csv(file)
        log("ROWS:", len(df))
        log("COLUMNS:", ", ".join(df.columns))

        if "status" in df:
            log("STATUS:", df["status"].value_counts().to_dict())

        groups = find_xyz_groups(df.columns)

        for group, axes in groups.items():
            xyz = df[
                [axes["x"], axes["y"], axes["z"]]
            ].apply(pd.to_numeric, errors="coerce").to_numpy()

            valid = np.isfinite(xyz).all(axis=1)

            if valid.any():
                log("XYZ GROUP:", group or "XYZ")
                log("  Valid:", int(valid.sum()))
                log("  Min:", np.round(
                    np.nanmin(xyz[valid], axis=0), 4
                ))
                log("  Max:", np.round(
                    np.nanmax(xyz[valid], axis=0), 4
                ))
                log("  Median:", np.round(
                    np.nanmedian(xyz[valid], axis=0), 4
                ))

        plot_xyz(df, file, label)

    except Exception as exc:
        log("ERROR:", repr(exc))

def inspect_npz(file):
    log("\nNPZ:", file)
    try:
        with np.load(file, allow_pickle=False) as data:
            for key in data.files:
                a = data[key]
                log(" ", key, a.shape, a.dtype)
    except Exception as exc:
        log("ERROR:", repr(exc))

# Wrist depth: original vs swapped.
for suffix in ("", "_swapped"):
    trial = FS / f"wristband_batch_{TAKE}{suffix}"

    wrist = trial / "wrist_xyz_camera1.csv"
    if wrist.exists():
        inspect_csv(
            wrist,
            "Swapped wrist camera XYZ" if suffix
            else "Original wrist camera XYZ"
        )

    # Reconstructed palm/finger outputs.
    for name in (
        "palm_pose_6dof.csv",
        "joint_targets.csv",
    ):
        for file in trial.rglob(name):
            inspect_csv(
                file,
                ("Swapped " if suffix else "Original ") + name
            )

# Discover MuJoCo simulation traces.
log("\n========== SIMULATION TRACE DISCOVERY ==========")

candidates = []

for base in (
    OUT / "reconstructed_demos",
    OUT / "diagnostics",
    OUT,
):
    if not base.exists():
        continue

    for file in base.rglob("*"):
        if not file.is_file():
            continue

        if file.suffix.lower() not in (".csv", ".npz"):
            continue

        name = file.name.lower()
        path = str(file).lower()

        if (
            "trace" in name
            or ("ik" in name and "csv" in name)
            or (TAKE in path and "sim" in path)
        ):
            candidates.append(file)

for file in sorted(set(candidates)):
    log("FOUND:", file)

    if file.suffix.lower() == ".csv":
        inspect_csv(file, "MuJoCo / IK trace")
    elif file.suffix.lower() == ".npz":
        inspect_npz(file)

# Inspect simulation script's trace-writing fields.
log("\n========== SIMULATOR TRACE CODE ==========")

script = ROOT / "code/dex5_video_to_sim_wrist_smoothed.py"

if script.exists():
    lines = script.read_text().splitlines()

    patterns = (
        "trace",
        "ik_error",
        "position_error",
        "target_xyz",
        "achieved_xyz",
        "DataFrame",
        "writerow",
    )

    hits = [
        i for i, line in enumerate(lines)
        if any(p in line.lower() for p in patterns)
    ]

    shown = set()

    for i in hits[:35]:
        for j in range(max(0, i-1), min(len(lines), i+2)):
            if j not in shown:
                log(f"{j+1}: {lines[j]}")
                shown.add(j)

report = DEST / "diagnostic_report.txt"
report.write_text("\n".join(REPORT) + "\n")

log("\nREPORT:", report)
log("PLOTS:", DEST)
