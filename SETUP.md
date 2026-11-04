# Launcher installation and first test

The files in this archive are an **additive overlay** for an existing `dex-5` checkout. They do not replace existing research scripts. Back up any same-named files first.

1. Extract this archive into the **dex-5 project root**, not the `code/` folder.
2. Activate your main `g1_lerobot` environment.
3. Verify configuration in `configs/default.yaml` (particularly calibration and SAM3 weights).
4. Execute `./real2sim --help` and `./real2sim doctor --take take_20261104_193136_730622`.
5. Preview the exact commands with `./real2sim process --take take_20261104_193136_730622 --dry-run` and `./real2sim replay --take take_20261104_193136_730622 --dry-run`.
6. Test the previously reconstructed take using `./real2sim replay --take take_20261104_193136_730622` (writes a **new** output directory, `outputs/real2sim_replay_<take>`).
7. For fresh input, `./real2sim record`, `./real2sim annotate --take <take>` and `./real2sim process --take <take>`.
8. The physics command uses `outputs/smoothed_<take>/sim_qpos.npz`, reflecting the current experimental physics script; it will report a missing dependency rather than silently choose a different trajectory.

## Environment notes

The primary `requirements.txt` is a conservative set of directly used or likely imported Python dependencies. It is not verified against every optional source file; running `pip install -r requirements.txt` in an already-working environment can unintentionally upgrade a functioning CUDA stack. Prefer preserving the working environment and use `python tools/capture_dependency_versions.py` first. FoundationStereo runs in its separate `.venv` and should be installed via its upstream documentation; `requirements-foundationstereo.txt` supplements (does not replace) upstream requirements.

## Notes about commands

`./real2sim record` forwards any extra arguments to the existing recorder. `annotate` checks the annotation script's `--help` for a supported recording flag before invoking it; it safely stops if the script's CLI is not recognized. In **annotate dry-run only**, the printed `--take` flag is provisional.

Run offline launcher tests with `python -m unittest discover -s tests -p 'test_real2sim_cli.py' -v`. Run from the dex-5 root. This tests launcher command construction, not GPU inference or physical grasp success.

To audit the **actual imports in your checkout**, run `python tools/audit_imports.py`. It will report third-party imports from the active files; send its output to review requirements before publishing. `python tools/capture_dependency_versions.py` reports installed version numbers without changing your environment. The repo's Python dependencies and system components (camera drivers, CUDA, ffmpeg, xclip) are separate requirements.

**Safety:** The launcher refuses to overwrite an existing processed take, existing `phase_annotations.csv`, or nonempty replay/physics output directories unless `--overwrite` is explicitly supplied. For a first real test, use the existing processed take for `replay` only; avoid rerunning GPU processing or annotations on that take.

## FoundationStereo Integration

Install FoundationStereo separately, then copy our custom processor:

    cp integrations/foundationstereo/dex5_wristband_pipeline_v2.py FoundationStereo/scripts/dex5_wristband_pipeline_v2.py
