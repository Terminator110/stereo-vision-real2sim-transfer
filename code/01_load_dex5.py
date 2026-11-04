from pathlib import Path
import mujoco
import mujoco.viewer

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Right Dex5 URDF
URDF_PATH = (
    ROOT
    / "models"
    / "dex5_1"
    / "Dex5-URDF-R"
    / "Dex5-URDF-R.urdf"
)

print("Loading:", URDF_PATH)

# Load URDF
model = mujoco.MjModel.from_xml_path(str(URDF_PATH))
data = mujoco.MjData(model)

print("\n=== MODEL INFORMATION ===")
print("Bodies:", model.nbody)
print("Joints:", model.njnt)
print("Actuators:", model.nu)
print("DOF:", model.nv)

print("\n=== JOINT NAMES ===")

for i in range(model.njnt):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    print(i, name)

# Initialize kinematics
mujoco.mj_forward(model, data)

print("\nOpening MuJoCo viewer...")

# Passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        viewer.sync()