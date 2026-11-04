#!/usr/bin/env python3
"""Export synchronized camera-1 / G1-DEX5 MuJoCo side-by-side video.

Run from the dex-5 project root with MUJOCO_GL=egl (NVIDIA) or osmesa.
Does not modify source recordings, robot assets, or trajectories.
"""
import argparse
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--take', required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--trajectory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--width', type=int, default=800, help='Width of EACH panel')
    parser.add_argument('--height', type=int, default=450, help='Height of EACH panel')
    parser.add_argument('--azimuth', type=float, default=135)
    parser.add_argument('--elevation', type=float, default=-22)
    parser.add_argument('--distance', type=float, default=1.8)
    parser.add_argument('--lookat', type=float, nargs=3, default=[0.25, -0.05, 0.95])
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    take = args.take
    rec = root / 'code/stereo/recordings_three_cam' / take / 'camera1.avi'
    sim = args.trajectory or root / 'outputs' / f'real2sim_replay_{take}' / 'sim_qpos.npz'
    out = args.output or root / 'outputs' / f'{take}_side_by_side.mp4'
    sim = sim.expanduser().resolve()
    out = out.expanduser().resolve()
    model_path = root / 'models/scenario_g1_dex5_stationary.xml'

    for file in (rec, sim, model_path):
        if not file.is_file():
            raise SystemExit(f'Missing required file: {file}')
    if out.exists():
        raise SystemExit(f'Refusing to overwrite existing video: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)

    with np.load(sim) as saved:
        qpos = saved['qpos']
        source_frames = saved['source_frames'].astype(int)
    if qpos.shape[0] != len(source_frames):
        raise SystemExit('Trajectory qpos/source_frames length mismatch')
    if np.any(np.diff(source_frames) < 0):
        raise SystemExit('source_frames must be sorted ascending')

    cap = cv2.VideoCapture(str(rec))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video: {rec}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < 1:
        fps = 10.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # The source AVI defines frame timing. For missing reconstruction frames,
    # hold the nearest available pose; do not compress recording time.
    if n_frames <= 0:
        n_frames = int(source_frames[-1]) + 1

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    if qpos.shape[1] != model.nq:
        raise SystemExit(f'Trajectory has {qpos.shape[1]} qpos columns; model needs {model.nq}')
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = args.lookat
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    camera.distance = args.distance

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (args.width * 2, args.height))
    if not writer.isOpened():
        raise SystemExit('Cannot open MP4 writer. Check OpenCV/codec support.')

    print(f'Camera video: {rec}', flush=True)
    print(f'MuJoCo trajectory: {sim}', flush=True)
    print(f'Output: {out}', flush=True)
    print(f'Source fps: {fps:.3f}, frames: {n_frames}', flush=True)
    try:
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            k = 0
            for frame_no in range(n_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                left = cv2.resize(frame, (args.width, args.height))

                # Hold the nearest available pose at sparse source frames.
                while (k + 1 < len(source_frames) and
                       abs(int(source_frames[k + 1]) - frame_no) <
                       abs(int(source_frames[k]) - frame_no)):
                    k += 1
                data.qpos[:] = qpos[k]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render()
                right = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                result = np.hstack((left, right))
                cv2.rectangle(result, (0, 0), (args.width * 2, 44), (20, 20, 20), -1)
                cv2.putText(result, 'REAL CAMERA 1', (18, 29),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(result, 'MUJOCO G1 + DEX5', (args.width + 18, 29),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                writer.write(result)
                if (frame_no + 1) % 50 == 0:
                    print(f'Exported {frame_no + 1}/{n_frames}', flush=True)
    finally:
        cap.release()
        writer.release()
    print(f'DONE: {out}', flush=True)


if __name__ == '__main__':
    main()
