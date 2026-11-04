# Dex-5 Native MuJoCo Test Baseline

Independent copy of the working 29-DOF G1 walking demo on a flat plane.
Dex5 hands have NOT been added yet; G1 still has fixed rubber hands.
No war-field scene, ROS, MCP, backend, or WASM is needed.

## Run

```bash
cd /home/sparx3d/physikal_workflow/dex-5
./walk --walk-s 4 --stand-s 3
```

`walk` uses `/usr/bin/python3`, not an activated conda environment, and
loads the copied policy cache offline. The native viewer requires a desktop display.
Pass other original script arguments unchanged, for example:

```bash
./walk --goto-x 1 --goto-y 0 --gait-mode walk
./walk --render false --realtime false --walk-s 4 --stand-s 3
```

The Python entry point is `code/g1_groot_walk_demo.py`. Its default model is
`models/scenario_g1_spawn_test.xml`; relative model paths resolve from this folder.
Existing named destinations are retained in the copied code, but are only coordinates
on this empty plane, not physical war-field sites.

## Layout

- `code/`: copied walking/controller integration script; only packaging paths changed.
- `models/`: flat-plane MJCF, G1 definitions/meshes, and copied GR00T policy cache.
- `vendor/lerobot/`: two required controller modules and upstream license.
- `vendor/python/`: copied local ONNX Runtime / Hugging Face dependencies.

System Python must still provide MuJoCo and NumPy (already available on this machine).
The copied native Python packages are machine/Python-version specific, not a portable venv.
Original UI worktrees and shared controller sources were not modified.
The original default elastic support remains enabled; this is not an unsupported-walking validation.

## Verification

Headless walk/stand smoke test passed with the bundled offline policies:
8.002 seconds of simulation, final pelvis position approximately (0.767, -0.063, 0.778) m.
The interactive viewer has not been opened as part of this copy operation.

## Demonstration

[Watch the Real2Sim demo](docs/media/real2sim_d

https://github.com/user-attachments/assets/b5b4f1de-79f5-4ff9-8da7-dcf48feeaf12

emo.mp4)

## Complete Workflow

[Step-by-step procedure](docs/WORKFLOW.md)
