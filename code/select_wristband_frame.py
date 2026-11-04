import cv2
from pathlib import Path

TAKE = Path.home() / (
    "physikal_workflow/dex-5/code/stereo/"
    "recordings_parallel_v3/"
    "take_20261103_211717_574941"
)

video = TAKE / "camera1.avi"

cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {video}")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame = 0

window = "SAM3 Wristband Prompt Frame Selector"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 1100, 750)

def change_frame(value):
    global frame
    frame = value

cv2.createTrackbar(
    "Frame", window, 0, total - 1, change_frame
)

print(f"Recording: {TAKE.name}")
print(f"Total frames: {total}")
print("Drag slider to browse frames.")
print("A/D: previous/next frame")
print("ENTER or S: select frame")
print("Q: cancel")

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()

    if not ok:
        break

    cv2.putText(
        img,
        f"Frame {frame}/{total-1} | ENTER: SELECT | A/D: navigate",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.imshow(window, img)

    key = cv2.waitKey(40) & 0xFF

    if key in (13, 10, ord('s')):
        output = TAKE / "selected_prompt_frame.txt"
        output.write_text(str(frame) + "\n")

        print(f"\nSELECTED FRAME: {frame}")
        print(f"Saved: {output}")
        break

    elif key in (ord('a'), 81):
        frame = max(0, frame - 1)
        cv2.setTrackbarPos("Frame", window, frame)

    elif key in (ord('d'), 83):
        frame = min(total - 1, frame + 1)
        cv2.setTrackbarPos("Frame", window, frame)

    elif key == ord('q'):
        print("Selection cancelled")
        break

cap.release()
cv2.destroyAllWindows()
