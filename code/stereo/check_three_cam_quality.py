import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Current device mapping
CAMERAS = {
    "Stereo 1": 4,
    "Stereo 2": 2,
    "Close-up 3": 0,
}

WIDTH, HEIGHT = 1280, 720
FPS = 30

caps = {}

for name, device in CAMERAS.items():
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {name}: /dev/video{device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    caps[name] = cap

    print(
        f"{name}: /dev/video{device} | "
        f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
        f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} | "
        f"Negotiated {cap.get(cv2.CAP_PROP_FPS):.1f} FPS"
    )

cv2.namedWindow("3 Camera Quality Check", cv2.WINDOW_NORMAL)
cv2.resizeWindow("3 Camera Quality Check", 1500, 900)

names = list(CAMERAS)
frames = {}

count = 0
start = time.monotonic()

def read_camera(name):
    return name, caps[name].read()

try:
    with ThreadPoolExecutor(max_workers=3) as pool:
        while True:
            results = list(pool.map(read_camera, names))

            for name, (ok, frame) in results:
                if ok:
                    frames[name] = frame

            if len(frames) != 3:
                continue

            panels = []

            for name in names:
                frame = frames[name]

                preview = cv2.resize(
                    frame, (640, 360),
                    interpolation=cv2.INTER_AREA
                )

                cv2.putText(
                    preview, name,
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2
                )

                panels.append(preview)

            top = cv2.hconcat(panels[:2])

            bottom = cv2.hconcat([
                panels[2],
                0 * panels[2]
            ])

            display = cv2.vconcat([top, bottom])

            cv2.putText(
                display,
                "S = save full-resolution images | Q = quit",
                (660, 410),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            cv2.imshow("3 Camera Quality Check", display)

            count += 1
            now = time.monotonic()

            if now - start >= 5:
                print(f"Combined preview FPS: {count/(now-start):.1f}")
                count = 0
                start = now

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                out = Path("/tmp/three_cam_quality")
                out.mkdir(exist_ok=True)

                for i, name in enumerate(names, 1):
                    filename = out / f"camera{i}.png"
                    cv2.imwrite(str(filename), frames[name])
                    print("Saved:", filename)

            elif key == ord("q"):
                break

finally:
    for cap in caps.values():
        cap.release()

    cv2.destroyAllWindows()
