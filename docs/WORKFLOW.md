# Stereo-Vision based Real2Sim Transfer — Reproduction Procedure

This guide documents the demonstrated **G1 + DEX5** pipeline: ChArUco calibration, synchronized video recording, SAM3 wristband segmentation, FoundationStereo depth, MediaPipe landmarks and DEX5 finger retargeting, MuJoCo kinematic replay, and optional experimental physics replay. **No VLA/RL training, HaMeR processing, or successful physical cup grasp is claimed.**

Run commands from the **`dex-5/` repository root**, unless explicitly noted.

## 1. Environment and dependency checks

Use two environments: the Conda `g1_lerobot` environment for camera tools, MediaPipe, retargeting and MuJoCo; and `FoundationStereo/.venv` for FoundationStereo inference. Install the requirements appropriate to each environment, and follow FoundationStereo's upstream instructions for its CUDA/PyTorch dependencies. Do not blindly upgrade a working CUDA installation.

```bash
cd /path/to/dex-5
conda activate g1_lerobot
chmod +x real2sim
./real2sim --help
./real2sim doctor --take TAKE_ID
FoundationStereo/.venv/bin/python -c 'import torch, omegaconf; print("FoundationStereo imports OK")'
```

`doctor` checks files and Python dependencies, **not** camera calibration accuracy or physical grasp success. `TAKE_ID` identifies an existing recording.

## 2. Identify cameras and preserve their orientation

```bash
v4l2-ctl --list-devices
for CAM in 0 2 4; do
    v4l2-ctl -d "/dev/video$CAM" --list-formats-ext
 done
```

The original stereo capture used 1280×720 images and USB cameras with USB ID `32e6:d412`. Device assignments can change between boots. Use the **same camera ordering, flips/rotations and resolution as calibration**. A third camera records a closer view but is not yet used by the automated finger-reconstruction pipeline.

## 3. Calibrate the stereo cameras

Capture ChArUco board image pairs in many positions, distances and tilts. The original calibration used approximately 36 pairs, pruned to 26. Start by inspecting your installed capture and calibration commands:

```bash
cd code
python stereo/charuco_capture.py --help
find stereo -type f \( -iname '*charuco*.py' -o -iname '*calib*.py' \) | sort
cd ..
```

Generate and preserve `code/stereo/calibration/parallel_results_v3/stereo_calibration.npz`. This file contains camera intrinsics/distortion, stereo rotation and translation, and rectification parameters. Use the actual options supported by the calibration script in your checkout; the supplied procedure does not establish a verified universal calibration command.

## 4. Calibrate the camera-to-table coordinate system

Generate and preserve `code/stereo/calibration/parallel_results_v3/table_frame_multi.npz`. The camera-to-table relation is conceptually:

```python
p_table = R_cam1_to_table @ p_cam1 + t_cam1_to_table
```

Use the calibration script corresponding to your camera setup. Do not copy the example table-origin and yaw settings to a different camera/table geometry without recalibration. The **tested** G1/MuJoCo mapping for this specific setup was yaw **−90°** and table world origin **(0.385, −0.120, 0.832) m**.

## 5. Record a demonstration

The normal launcher command is:

```bash
./real2sim record
```

The documented original recorder invocation, useful when camera IDs must be specified explicitly, was:

```bash
cd code
python stereo/record_three_cameras.py \
    --cam1 4 --cam2 2 --cam3 0 \
    --width 1280 --height 720 --fps 10 \
    --rotate-cam2 --rotate-cam3
cd ..
```

**Space** starts/stops recording; **Q** quits. New takes are stored in `code/stereo/recordings_three_cam/take_YYYYMMDD_HHMMSS_xxxxxx/` with three AVI files, `frame_timestamps.csv` and `recording_meta.json`. Inspect the launcher command with `./real2sim record --dry-run` before a new camera session: the default launcher command may not contain the camera flags above.

## 6. Annotate manipulation phases

```bash
TAKE=take_20261104_193136_730622
./real2sim annotate --take "$TAKE"
```

Equivalent underlying script:

```bash
python code/annotate_phases.py --video "code/stereo/recordings_three_cam/$TAKE/camera1.avi"
```

Mark **approach**, **grasp**, **transport**, **release**, and **retract** using the annotator's displayed controls. The annotator outputs `phase_annotations.csv` and `.json` into the recording. The approach-end and grasp-end frames are used by experimental cup placement and physics. **Do not overwrite existing annotations unintentionally.**

## 7. Prepare timestamps for three-camera recordings

The three-camera recorder generates `frame_timestamps.csv`; the older processing stages may expect `timestamps.csv`. Before processing a new recording, check whether the current launcher handles this conversion. If it does not, use the following non-overwriting conversion when the expected source columns are present:

```bash
python - "$TAKE" <<'PY'
import csv, sys
from pathlib import Path
rec = Path('code/stereo/recordings_three_cam') / sys.argv[1]
src, dst = rec / 'frame_timestamps.csv', rec / 'timestamps.csv'
if dst.exists():
    print('Existing timestamps preserved:', dst)
else:
    with src.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError('No source timestamps')
    needed = {'frame', 'sample_time_ns', 'camera1_time_ns', 'camera2_time_ns'}
    if not needed.issubset(rows[0]):
        raise RuntimeError('Unexpected timestamp columns: ' + str(list(rows[0])))
    t0 = int(rows[0]['sample_time_ns'])
    with dst.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('frame', 'time_s', 'cam1_monotonic_ns', 'cam2_monotonic_ns'))
        for r in rows:
            w.writerow((r['frame'], (int(r['sample_time_ns'])-t0)/1e9,
                        r['camera1_time_ns'], r['camera2_time_ns']))
    print('Created', dst, 'for', len(rows), 'frames')
PY
```

## 8. Select wristband and run reconstruction

Use `code/stereo/select_and_reconstruct.py` to inspect/select a wristband frame where it is clearly visible; in the earlier workflow the selector used A/D to navigate and S to select. Check its local `--help` before invoking it.

```bash
python code/stereo/select_and_reconstruct.py --help
./real2sim process --take "$TAKE" --dry-run
./real2sim process --take "$TAKE"
```

The launcher orchestrates:

1. Stereo image preparation.
2. **SAM3** wristband masks (`code/dex5_sam3_video_masks.py`).
3. **FoundationStereo** depth (`FoundationStereo/scripts/dex5_wristband_pipeline_v2.py`).
4. Stereo MediaPipe hand detection and finger retargeting (`code/12_stereo_g1_dex5.py`).
5. Combined palm position/orientation and DEX5 joint targets.
6. Optional cup/coaster localization.

The FoundationStereo stage must use `FoundationStereo/.venv/bin/python`, not an unrelated IsaacLab environment. A previous run encountered that interpreter mismatch; inspect the printed command if Stage 03 fails.

If the complete run already exists, **avoid forcing regeneration**. Outputs include `FoundationStereo/wristband_batch_$TAKE/palm_pose_6dof.csv` and `joint_targets.csv`.

## 9. Replay the human trajectory in MuJoCo

```bash
./real2sim replay --take "$TAKE" --dry-run
./real2sim replay --take "$TAKE"
```

For noninteractive execution:

```bash
./real2sim replay --take "$TAKE" --headless
```

In the validated setup, the launcher provides `--table-yaw-deg -90`, `--table-origin-world 0.385 -0.120 0.832`, and `--ik-mode xyz` to `code/dex5_video_to_sim_side_by_side.py`. Output is saved to `outputs/real2sim_replay_$TAKE/` (`sim_qpos.npz`, `sim_quality.csv`, `sim_report.json`). The position-first replay does **not** accurately track palm orientation and is **kinematic only**, without contacts, collision avoidance or grasp validation.

## 10. Optional experimental physics replay

First verify that replay output exists. In Launcher v3, the physics script can accept that trajectory explicitly, and its interface can be previewed with:

```bash
./real2sim physics --take "$TAKE" --dry-run
./real2sim physics --take "$TAKE" \
  --output "outputs/real2sim_physics_$TAKE"
```

The experimental cup-placement rule uses the MuJoCo hand-palm marker at the annotated **approach end**, rotates the marker −30° about its local X/red axis, projects the marker's positive Z/blue direction onto the table plane, then places the upright cup at **cup collision radius + 70 mm** along that direction, with its base on the table. The script generates a dynamic scene, simulates contacts and optional wrist lift, and saves a report. This placement is a test convention rather than a measured cup pose.

**Validation limit:** In the tested take, the physics script completed 12,702 steps but recorded zero finger and thumb contacts during hold, zero cup rise, and unsuccessful grasp. This must not be represented as a successful physical grasp transfer.

## 11. Evaluation and current results

For `take_20261104_193136_730622`, the tested XYZ kinematic replay reported **232 frames**, **19 DEX5 finger-joint bindings**, 200 `TRACKING` frames, seven `APPROX` frames and 25 `LARGE_ERROR` frames. The median **IK target residual** was 0.001 mm—**not** ground-truth hand-position accuracy. The system had measured palm orientation for 106 frames, and XYZ-only replay had median orientation error of 133.3°. These figures describe this take and configuration only.

Inspect `sim_quality.csv`, `sim_report.json` and the synchronized video preview before interpreting a result. Palm orientation and robust physical grasping remain unresolved research limitations.

## 12. Files needed in the repository

Retain the launcher, configs, Python source modules and their dynamic imports, G1/DEX5 MuJoCo XML, referenced meshes and objects, calibration scripts, the **reference** stereo/table calibration `.npz` files where redistribution is appropriate, requirements and this guide. Review upstream G1/DEX5 asset licenses. Exclude large recordings, model weights, FoundationStereo upstream checkout, environments and generated outputs. Do **not** delete local data as part of Git cleanup.

The repository should be initialized **inside `dex-5/`** and never staged from its `physikal_workflow/` parent. Check `git rev-parse --show-toplevel` before staging.
