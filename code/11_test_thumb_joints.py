"""Isolate all four right-Dex5 thumb joints, one at a time; no camera."""
import argparse
import csv
from datetime import datetime
import importlib.util
from pathlib import Path
import threading
import time
import mujoco
import numpy as np
from thumb_retarget import THUMB_NAMES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('retarget',Path(__file__).with_name('07_retarget_dex5_right.py'))
retarget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retarget)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/datetime.now().strftime('thumb_joint_test_%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(retarget.DEFAULT_MODEL))
    data = mujoco.MjData(model)
    bindings = retarget.joint_bindings(model)
    args.output.mkdir(parents=True,exist_ok=False)
    viewer,threads = None,[]
    try:
        if not args.headless:
            from mujoco import viewer as mj_viewer
            existing = set(threading.enumerate())
            viewer = mj_viewer.launch_passive(model,data)
            threads = [t for t in threading.enumerate() if t not in existing]
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance = max(0.25,model.stat.extent*1.6)
        with (args.output/'thumb_sweep.csv').open('w',newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['active_joint','value_rad',*THUMB_NAMES])
            for name in THUMB_NAMES:
                address,low,high = bindings[name]
                print(f'Testing {name}: {low:.3f} .. {high:.3f} radians',flush=True)
                for alpha in np.linspace(0,1,91):
                    if viewer is not None and not viewer.is_running():
                        return
                    begin = time.monotonic()
                    pose = model.qpos0.copy()
                    pose[address] = np.interp(alpha,[0,1/3,2/3,1],[0,low,high,0])
                    retarget.apply_pose(model,data,pose)
                    writer.writerow([name,pose[address],*[pose[bindings[n][0]] for n in THUMB_NAMES]])
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(max(0,1/30-(time.monotonic()-begin)))
        print(f'All four thumb joint sweeps complete; {args.output}',flush=True)
    finally:
        if viewer is not None:
            viewer.close()
            for thread in threads:
                thread.join(timeout=5)


if __name__ == '__main__':
    main()
