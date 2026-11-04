from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    ROOT
    / "models"
    / "dex5_1"
    / "Dex5-URDF-R"
    / "Dex5-URDF-R.urdf"
)


# ==========================================
# LOAD MODEL
# ==========================================

model = mujoco.MjModel.from_xml_path(str(URDF_PATH))
data = mujoco.MjData(model)

print("\n=== AVAILABLE JOINTS ===\n")

joint_ids = []

for i in range(model.njnt):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    joint_type = model.jnt_type[i]

    # Only single-DoF joints
    if joint_type not in (
        mujoco.mjtJoint.mjJNT_HINGE,
        mujoco.mjtJoint.mjJNT_SLIDE
    ):
        continue

    low, high = model.jnt_range[i]

    print(
        f"ID: {i:2d} | "
        f"Name: {name:25s} | "
        f"Range: [{low:.3f}, {high:.3f}]"
    )

    joint_ids.append(i)


# ==========================================
# SELECT JOINT
# ==========================================

# Change this after inspecting printed joints
JOINT_ID = joint_ids[0]

joint_name = mujoco.mj_id2name(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    JOINT_ID
)

qpos_index = model.jnt_qposadr[JOINT_ID]

low, high = model.jnt_range[JOINT_ID]

print("\nSelected joint:", joint_name)
print("Joint limits:", low, high)


# ==========================================
# INITIALIZE
# ==========================================

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)


# ==========================================
# ANIMATION
# ==========================================

FREQUENCY = 0.25  # Hz

with mujoco.viewer.launch_passive(model, data) as viewer:

    start_time = time.monotonic()

    while viewer.is_running():

        t = time.monotonic() - start_time

        # Smooth oscillation between limits
        alpha = 0.5 * (
            1.0 + np.sin(2.0 * np.pi * FREQUENCY * t)
        )

        target = low + alpha * (high - low)

        # Directly set joint position
        data.qpos[qpos_index] = target

        # Update forward kinematics
        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(0.01)