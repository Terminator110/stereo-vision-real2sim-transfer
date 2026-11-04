import argparse
import csv
import json
import time
from pathlib import Path
import cv2

PHASES = ["approach", "grasp", "transport", "release", "retract"]

p = argparse.ArgumentParser()
p.add_argument("--video", required=True)
args = p.parse_args()

video = Path(args.video).expanduser().resolve()
cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise SystemExit(f"Cannot open: {video}")

n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
encoded_fps = cap.get(cv2.CAP_PROP_FPS)
if n <= 0:
    raise SystemExit("Cannot determine video frame count")

output = video.parent / "phase_annotations.csv"
json_output = video.parent / "phase_annotations.json"

frame = 0
playing = False
phase_idx = 0
start_frame = None
annotations = []

window = "PHASE ANNOTATOR | S=start E=end A/D=step SPACE=play Q=quit"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 1280, 800)

def seek(value):
    global frame
    frame = value

cv2.createTrackbar("Frame", window, 0, n - 1, seek)

def save():
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["phase", "start_frame", "end_frame"]
        )
        writer.writeheader()
        writer.writerows(annotations)

    json_output.write_text(json.dumps({
        "video": str(video),
        "total_frames": n,
        "encoded_fps": encoded_fps,
        "phases": annotations
    }, indent=2))

def mark_start():
    global start_frame
    if phase_idx >= len(PHASES):
        print("All phases completed.")
        return
    if start_frame is not None:
        print("Finish the current phase with E first.")
        return
    if annotations and frame <= annotations[-1]["end_frame"]:
        print("Start must follow the previous phase end.")
        return
    start_frame = frame
    print(f"START {PHASES[phase_idx]}: frame {frame}")

def mark_end():
    global start_frame, phase_idx
    if phase_idx >= len(PHASES) or start_frame is None:
        print("Press S to mark the phase start first.")
        return
    if frame < start_frame:
        print("End cannot be before start.")
        return

    row = {
        "phase": PHASES[phase_idx],
        "start_frame": start_frame,
        "end_frame": frame,
    }
    annotations.append(row)
    print(f"SAVED: {row}")
    phase_idx += 1
    start_frame = None
    save()
    if phase_idx == len(PHASES):
        print("ALL FIVE PHASES SAVED.")

print("PHASE ORDER:", " -> ".join(PHASES))
print("S=start E=end A/D=step SPACE=play/pause Q=save+quit")
print("Annotations:", output)

last = time.monotonic()

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    if not ok:
        break

    h, w = img.shape[:2]
    display = cv2.copyMakeBorder(
        img, 0, 85, 0, 0, cv2.BORDER_CONSTANT,
        value=(25, 25, 25)
    )

    current = (
        PHASES[phase_idx].upper()
        if phase_idx < len(PHASES)
        else "COMPLETE"
    )

    info1 = f"FRAME {frame}/{n-1}   PHASE: {current}"
    info2 = (
        f"START: {start_frame if start_frame is not None else '-'}"
        "   S=start  E=end  A/D=step  SPACE=play  Q=save"
    )

    cv2.putText(display, info1, (15, h+32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255,255,255), 2)
    cv2.putText(display, info2, (15, h+66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180,255,180), 1)

    cv2.imshow(window, display)

    key = cv2.waitKeyEx(15)
    now = time.monotonic()

    if key in (ord("q"), ord("Q"), 27):
        break
    elif key in (ord("s"), ord("S")):
        playing = False
        mark_start()
    elif key in (ord("e"), ord("E")):
        playing = False
        mark_end()
    elif key in (ord("a"), ord("A"), 2424832):
        playing = False
        frame = max(0, frame-1)
    elif key in (ord("d"), ord("D"), 2555904):
        playing = False
        frame = min(n-1, frame+1)
    elif key == 32:
        playing = not playing
        last = now

    if playing and now - last >= 0.1:
        frame = min(frame+1, n-1)
        last = now
        if frame == n-1:
            playing = False

    cv2.setTrackbarPos("Frame", window, frame)

save()
cap.release()
cv2.destroyAllWindows()
print(f"\nSaved {len(annotations)}/5 phases")
print(output)
print(json_output)
