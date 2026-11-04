#!/usr/bin/env python3
"""Two-camera preview. Camera 2 is vertically flipped FOR DISPLAY ONLY."""
import cv2
import time

CAM1 = 2
CAM2 = 0
WIDTH, HEIGHT, FPS = 1280, 720, 30
FLIP_CAM2_PREVIEW = False  # Flip around horizontal axis (top <-> bottom)


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"Camera {index}: {actual[0]}x{actual[1]} @ {cap.get(cv2.CAP_PROP_FPS):.1f} FPS")
    if actual != (WIDTH, HEIGHT):
        cap.release()
        raise RuntimeError(f"Camera {index} returned {actual}, expected {(WIDTH, HEIGHT)}")
    return cap


def main():
    cam1 = cam2 = None
    try:
        cam1 = open_camera(CAM1)
        cam2 = open_camera(CAM2)
        t0 = time.monotonic()
        count = 0
        while True:
            a = cam1.grab()
            b = cam2.grab()
            if not (a and b):
                print("Camera grab failed; check USB connection")
                break
            a, frame1 = cam1.retrieve()
            b, frame2 = cam2.retrieve()
            if not (a and b):
                print("Camera retrieve failed")
                break

            # Rotate Camera 2 image by 180 degrees (both flips).
            frame2 = cv2.flip(frame2, -1)

            view1 = cv2.resize(frame1, (640, 360))
            view2 = cv2.resize(frame2, (640, 360))
            cv2.putText(view1, "CAMERA 1", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
            cv2.putText(view2, "CAMERA 2 (DISPLAY FLIP)", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
            cv2.imshow("Stereo Preview - Q to quit", cv2.hconcat([view1, view2]))
            count += 1
            if time.monotonic() - t0 >= 3:
                print(f"Preview loop: {count/(time.monotonic()-t0):.1f} FPS")
                t0, count = time.monotonic(), 0
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        if cam1 is not None:
            cam1.release()
        if cam2 is not None:
            cam2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
