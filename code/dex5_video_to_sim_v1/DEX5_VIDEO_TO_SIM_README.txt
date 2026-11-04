DEX-5: new stereo recording -> G1 + Dex5 kinematic MuJoCo preview
=============================================================

This replay pipeline uses two already-existing extraction stages:
(A) wristband_trial_new/palm_pose_6dof.csv from SAM3 + FoundationStereo.
(B) 10_g1_dex5_stationary.py --video --no-wrist to generate verified Dex5 finger joint targets.
It merges (A) and (B) and solves right-arm IK in MuJoCo using dex5_video_to_sim_v1.py.

Annotation: NOT required for the first kinematic replay. For later imitation
learning, annotate Approach, Grasp, Lift, Transport, Place/Release, Retreat,
and trial success; use labels from the NEW recording, not annotation1.json
from the old November 2 take.

Install dex5_video_to_sim_v1.py in:
  ~/physikal_workflow/dex-5/code/dex5_video_to_sim_v1.py

Extract Dex5 finger targets (existing retargeter; source take from current trial):
  cd ~/physikal_workflow/dex-5/code
  conda activate g1_lerobot
  TAKE=$(python - <<'PY'
import json
from pathlib import Path
p=Path.home()/'physikal_workflow/dex-5/FoundationStereo/wristband_trial_new/meta.json'
print(json.loads(p.read_text())['take'])
PY
)
  FINGER_OUT=../outputs/dex5_fingers_$(date +%Y%m%d_%H%M%S)
  python 10_g1_dex5_stationary.py --video "$TAKE/camera1.avi" --no-wrist --headless --max-frames 150 --output "$FINGER_OUT"
  echo "Finger targets: $FINGER_OUT/joint_targets.csv"

Test only 30 frames, arm follows wrist XYZ; fingers follow existing retarget:
  MUJOCO_GL=glfw python dex5_video_to_sim_v1.py \
    --palm-csv ../FoundationStereo/wristband_trial_new/palm_pose_6dof.csv \
    --finger-csv "$FINGER_OUT/joint_targets.csv" \
    --ik-mode xyz --end 29 \
    --output ../outputs/dex5_sim_xyz_30

Full 150-frame pose test (absolute human palm basis, source invalid frames held):
  MUJOCO_GL=glfw python dex5_video_to_sim_v1.py \
    --palm-csv ../FoundationStereo/wristband_trial_new/palm_pose_6dof.csv \
    --finger-csv "$FINGER_OUT/joint_targets.csv" \
    --ik-mode pose --output ../outputs/dex5_sim_pose_150

Outputs: sim_quality.csv, sim_qpos.npz, sim_report.json.
First-position anchor is DIAGNOSTIC, not a calibrated table-to-MuJoCo world
translation; use --table-origin-world X Y Z after that registration is known.
No physics step, collisions or cup attachment in this demo. Palm orientation
is unavailable in about 40/150 frames of this trial, and is held there.
