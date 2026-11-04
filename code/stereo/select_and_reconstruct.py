#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--take", required=True)
args = parser.parse_args()

ROOT = Path.home() / "physikal_workflow/dex-5"
CODE = ROOT / "code"

TAKE = args.take
REC = CODE / "stereo/recordings_three_cam" / TAKE
VIDEO = REC / "camera1.avi"

if not VIDEO.exists():
    raise SystemExit(f"Missing video: {VIDEO}")

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit("Cannot open video")

N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame = 0
selected = None

WIN = "SELECT WRISTBAND FRAME"

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 800)


def seek(value):
    global frame
    frame = value


cv2.createTrackbar("Frame", WIN, 0, N - 1, seek)

print("\n========== FRAME SELECTOR ==========")
print("A / Left Arrow  : Previous frame")
print("D / Right Arrow : Next frame")
print("S               : Select frame and reconstruct")
print("Q               : Quit")
print("====================================\n")

while True:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()

    if not ok:
        break

    cv2.putText(
        img,
        f"FRAME {frame} / {N-1}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow(WIN, img)

    key = cv2.waitKeyEx(0)

    if key in (ord("q"), ord("Q"), 27):
        break

    elif key in (ord("a"), ord("A"), 2424832):
        frame = max(0, frame - 1)

    elif key in (ord("d"), ord("D"), 2555904):
        frame = min(N - 1, frame + 1)

    elif key in (ord("s"), ord("S")):
        selected = frame
        break

    # Correct OpenCV API:
    # setTrackbarPos(trackbar_name, window_name, position)
    cv2.setTrackbarPos("Frame", WIN, frame)

cap.release()
cv2.destroyAllWindows()

if selected is None:
    print("Selection cancelled.")
    sys.exit(0)

(REC / "selected_prompt_frame.txt").write_text(
    f"{selected}\n"
)

print(f"\nSELECTED FRAME: {selected}")
print("Starting reconstruction...\n")

FS_PY = (
    ROOT / "FoundationStereo/.venv/bin/python"
)

command = [
    sys.executable,
    str(CODE / "dex5_batch_pipeline_v2.py"),
    "process",
    "--take", str(REC),
    "--prompt-frame", str(selected),
    "--sam-weights", str(
        Path.home() / "physikal_workflow/sam3.pt"
    ),
    "--g1-python", sys.executable,
    "--fs-python", str(FS_PY),
    "--scale", "0.5",
    "--iters", "16",
    "--max-reproj", "8",
    "--force",
]

print("COMMAND:")
print(" ".join(command))

subprocess.run(command, cwd=CODE, check=True)
