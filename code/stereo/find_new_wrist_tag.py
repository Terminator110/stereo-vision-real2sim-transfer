import cv2
import sys
import time
from collections import Counter

KNOWN_IDS = {4, 5}
CAM1, CAM2 = 2, 0

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, params)

caps = []
counts = Counter()

try:
    for index in (CAM1, CAM2):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{index}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        caps.append(cap)

    print(
        "Show all three wrist tags to the cameras. Scanning for 10 seconds...",
        file=sys.stderr
    )

    start = time.monotonic()

    while time.monotonic() - start < 10:
        caps[0].grab()
        caps[1].grab()

        for camera, cap in enumerate(caps):
            ok, frame = cap.retrieve()
            if not ok:
                continue

            if camera == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            _, ids, _ = detector.detectMarkers(frame)

            if ids is not None:
                for tag_id in set(int(x) for x in ids.flatten()):
                    if tag_id not in KNOWN_IDS:
                        counts[tag_id] += 1

    candidates = [tag_id for tag_id, n in counts.items() if n >= 3]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one new tag ID; detected candidates: {dict(counts)}"
        )

    new_id = candidates[0]
    print(f"New tag identified: {new_id}", file=sys.stderr)

    # stdout contains only the numeric ID for the shell.
    print(new_id)

finally:
    for cap in caps:
        cap.release()
