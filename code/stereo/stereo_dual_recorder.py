#!/usr/bin/env python3
"""Record paired videos from two calibrated USB cameras.

R: start/stop a recording. Q/ESC: finish and exit.
Raw images (no display flips) are written for downstream stereo processing.
Each session has frame timestamps and a capture-info text file.
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if size != (width, height):
        cap.release()
        raise RuntimeError(f"/dev/video{index} returned {size}; expected {(width, height)}")
    return cap


def writer(path, fps, width, height):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Unable to create {path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=int, default=2)
    p.add_argument("--right", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=20.0, help="Encoded playback FPS; use actual sustained capture FPS (default 20)")
    p.add_argument("--camera-fps", type=float, default=30.0, help="Requested camera FPS")
    p.add_argument("--out", type=Path, default=Path("stereo/recordings"))
    p.add_argument("--flip-right-preview", action="store_true", default=True)
    p.add_argument("--no-flip-right-preview", action="store_false", dest="flip_right_preview")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    left = right = None
    writers = None
    timestamp_file = None
    csvwriter = None
    session = None
    frames = 0
    started = None
    prev_time = None
    intervals = []
    capture_count = 0

    def stop():
        nonlocal writers, timestamp_file, csvwriter, session, started, prev_time, frames, intervals
        if writers is None:
            return
        for w in writers:
            w.release()
        writers = None
        timestamp_file.close()
        timestamp_file = None
        csvwriter = None
        elapsed = time.monotonic() - started
        measured = (frames - 1) / (sum(intervals)) if len(intervals) > 0 and sum(intervals) > 0 else 0
        msg = (f"Frames per camera: {frames}\nWall-clock duration: {elapsed:.3f} s\n"
               f"Average capture FPS: {measured:.3f}\nEncoded playback FPS: {args.fps:.3f}\n"
               f"Playback duration: {frames / args.fps:.3f} s\n"
               f"WARNING: videos play at fixed encoded FPS; use timestamps.csv for real timing.\n")
        (session / "summary.txt").write_text(msg)
        print(f"\nSAVED: {session}\n{msg}", flush=True)
        session = None
        started = None
        prev_time = None
        frames = 0
        intervals = []

    try:
        left = camera(args.left, args.width, args.height, args.camera_fps)
        right = camera(args.right, args.width, args.height, args.camera_fps)
        print(f"Camera 1 /dev/video{args.left}, Camera 2 /dev/video{args.right}")
        print("R = start / stop paired recordings; Q or Esc = quit")
        print("Files are RAW/unflipped. Camera 2 preview may be flipped.")
        print(f"Video files are encoded at {args.fps:g} FPS. Real timing goes to timestamps.csv.")
        while True:
            g1 = left.grab()
            g2 = right.grab()
            if not (g1 and g2):
                print("Camera grab failed; stopping.")
                break
            ok1, a = left.retrieve()
            time1 = time.monotonic_ns()
            ok2, b = right.retrieve()
            time2 = time.monotonic_ns()
            if not (ok1 and ok2):
                print("Frame retrieval failed; stopping.")
                break
            capture_count += 1

            if writers is not None:
                writers[0].write(a)
                writers[1].write(b)
                frames += 1
                csvwriter.writerow([frames - 1, f"{(time1 / 1e9 - started):.6f}", f"{(time2 / 1e9 - started):.6f}", time1, time2])
                t = (time1 + time2) / 2e9
                if prev_time is not None:
                    intervals.append(t - prev_time)
                prev_time = t
                if frames % 100 == 0:
                    timestamp_file.flush()

            preview1 = cv2.resize(a, (640, 360))
            preview2 = cv2.resize(b, (640, 360))
            if args.flip_right_preview:
                preview2 = cv2.flip(preview2, 0)
            preview = np.hstack([preview1, preview2])
            status = f"{'REC' if writers else 'READY'} | R=start/stop Q=quit"
            if writers:
                status += f" | {frames} frames | {time.monotonic() - started:.1f}s"
            cv2.rectangle(preview, (0, 0), (1279, 42), (0, 0, 0), -1)
            cv2.putText(preview, status, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                        (0, 0, 255) if writers else (0, 255, 0), 2)
            cv2.imshow("Stereo dual camera recorder", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                if writers is not None:
                    stop()
                else:
                    label = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    session = args.out / f"take_{label}"
                    session.mkdir(parents=True, exist_ok=False)
                    try:
                        writers = (writer(session / "camera1.avi", args.fps, args.width, args.height),
                                   writer(session / "camera2.avi", args.fps, args.width, args.height))
                        timestamp_file = (session / "timestamps.csv").open("w", newline="")
                        csvwriter = csv.writer(timestamp_file)
                        csvwriter.writerow(["frame", "cam1_retrieved_elapsed_s", "cam2_retrieved_elapsed_s",
                                            "cam1_retrieved_monotonic_ns", "cam2_retrieved_monotonic_ns"])
                        started = time.monotonic()
                        frames = 0
                        intervals = []
                        prev_time = None
                        print(f"\nRECORDING: {session}", flush=True)
                    except Exception:
                        if writers is not None:
                            for w in writers:
                                w.release()
                        writers = None
                        if timestamp_file is not None:
                            timestamp_file.close()
                            timestamp_file = None
                        raise
    finally:
        stop()
        if left is not None:
            left.release()
        if right is not None:
            right.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
