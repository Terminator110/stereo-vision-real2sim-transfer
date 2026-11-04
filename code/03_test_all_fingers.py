from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    ROOT
    / "models"
    / "dex5_1"
    / "Dex5-URDF-R"
    / "Dex5-URDF-R.urdf"
)

FPS = 60

# Dex5 finger groups
# Based on official Unitree URDF joint names.
FINGERS = {
    "thumb": [
        "Yaw_11R",
        "Roll_12R",
        "Pitch_13R",
        "Pitch_14R",
    ],
    "index": [
        "Roll_21R",
        "Pitch_22R",
        "Pitch_23R",
        "Pitch_24R",
    ],
    "middle": [
        "Roll_31R",
        "Pitch_32R",
        "Pitch_33R",
        "Pitch_34R",
    ],
    "ring": [
        "Roll_41R",
        "Pitch_42R",
        "Pitch_43R",
        "Pitch_44R",
    ],
    "pinky": [
        "Roll_51R",
        "Pitch_52R",
        "Pitch_53R",
        "Pitch_54R",
    ],
}


# ==========================================
# LOAD MODEL
# ==========================================

model = mujoco.MjModel.from_xml_path(str(URDF_PATH))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)


# ==========================================
# BUILD JOINT MAPPING
# ==========================================

finger_joints = {}

for finger_name, joint_names in FINGERS.items():

    finger_joints[finger_name] = []

    for joint_name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        if joint_id == -1:
            raise ValueError(f"Joint not found: {joint_name}")

        qpos_id = model.jnt_qposadr[joint_id]

        low, high = model.jnt_range[joint_id]

        finger_joints[finger_name].append({
            "name": joint_name,
            "qpos_id": qpos_id,
            "low": low,
            "high": high,
        })

print("\n=== DEX5 FINGER JOINTS ===")

for name, joints in finger_joints.items():

    print(f"\n{name.upper()}")

    for joint in joints:
        print(
            joint["name"],
            joint["low"],
            joint["high"]
        )


# ==========================================
# CONTROL FUNCTIONS
# ==========================================

def set_finger(finger_name, curl):

    """
    curl = 0.0 -> open
    curl = 1.0 -> closed

    For this test, the finger's flexion
    joints move together.
    """

    curl = np.clip(curl, 0.0, 1.0)

    joints = finger_joints[finger_name]

    for joint in joints:

        # Keep roll/yaw at zero.
        # Test only finger bending.
        if joint["name"].startswith(("Roll", "Yaw")):
            continue

        low = joint["low"]
        high = joint["high"]

        target = low + curl * (high - low)

        data.qpos[joint["qpos_id"]] = target


def open_hand():

    for finger in FINGERS:
        set_finger(finger, 0.0)


def close_hand():

    for finger in FINGERS:
        set_finger(finger, 1.0)


def interpolate(start, end, duration, update):

    """Smooth transition between two hand poses."""

    steps = max(1, int(duration * FPS))

    for i in range(steps + 1):

        alpha = i / steps

        # Smoothstep
        alpha = alpha * alpha * (3 - 2 * alpha)

        value = start + alpha * (end - start)

        update(value)

        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(1 / FPS)


# ==========================================
# ANIMATION
# ==========================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    open_hand()
    mujoco.mj_forward(model, data)
    viewer.sync()

    # Move each finger individually
    for finger in FINGERS:

        if not viewer.is_running():
            break

        print(f"\nMoving {finger}...")

        interpolate(
            0.0,
            1.0,
            1.0,
            lambda x, f=finger: set_finger(f, x)
        )

        interpolate(
            1.0,
            0.0,
            1.0,
            lambda x, f=finger: set_finger(f, x)
        )

    # Open/close all fingers
    while viewer.is_running():

        print("\nClosing hand...")

        interpolate(
            0.0,
            1.0,
            2.0,
            lambda x: [
                set_finger(f, x)
                for f in FINGERS
            ]
        )

        print("\nOpening hand...")

        interpolate(
            1.0,
            0.0,
            2.0,
            lambda x: [
                set_finger(f, x)
                for f in FINGERS
            ]
        )