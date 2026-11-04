#!/usr/bin/env python3

import argparse
import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--cam1", type=int, default=2)
parser.add_argument("--cam2", type=int, default=0)
parser.add_argument("--cam3", type=int, default=4)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument("--rotate-cam2", action="store_true")
parser.add_argument("--rotate-cam3", action="store_true")
parser.add_argument(
    "--output",
    default="stereo/recordings_three_cam"
)
args = parser.parse_args()

if args.fps <= 0:
    raise SystemExit("FPS must be positive")

if len({args.cam1, args.cam2, args.cam3}) != 3:
    raise SystemExit("Camera devices must be different")


class Camera:
    def __init__(self, device, name, rotate=False):
        self.name = name
        self.device = device
        self.rotate = rotate

        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{device}")

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.cap.set(cv2.CAP_PROP_FPS, args.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.timestamp_ns = None
        self.sequence = 0
        self.running = True

        self.thread = threading.Thread(
            target=self.capture_loop,
            daemon=True
        )
        self.thread.start()

    def capture_loop(self):
        while self.running:
            ok, frame = self.cap.read()

            timestamp = time.monotonic_ns()

            if not ok:
                time.sleep(0.01)
                continue

            if self.rotate:
                frame = cv2.rotate(
                    frame,
                    cv2.ROTATE_180
                )

            with self.lock:
                self.frame = frame
                self.timestamp_ns = timestamp
                self.sequence += 1

    def get(self):
        with self.lock:
            if self.frame is None:
                return None

            return (
                self.frame.copy(),
                self.timestamp_ns,
                self.sequence
            )

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


cameras = [
    Camera(args.cam1, "camera1"),
    Camera(args.cam2, "camera2", args.rotate_cam2),
    Camera(args.cam3, "camera3", args.rotate_cam3),
]

print("\n========== CAMERAS ==========")

for c in cameras:
    print(f"{c.name}: /dev/video{c.device}")

print("\nSPACE = Start / Stop recording")
print("Q = Quit")
print("All three streams are saved with shared frame indices.")
print("Individual capture timestamps are also preserved.\n")


def make_preview(frames, recording, count, skew_ms):
    panels = []

    for i, frame in enumerate(frames):
        panel = cv2.resize(frame, (640, 360))

        cv2.putText(
            panel,
            f"CAMERA {i+1}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        panels.append(panel)

    top = np.hstack([panels[0], panels[1]])

    bottom = np.zeros_like(top)
    bottom[:360, :640] = panels[2]

    output = np.vstack([top, bottom])

    status = (
        f"RECORDING | Frame {count}"
        if recording else
        "PREVIEW | SPACE to record"
    )

    cv2.putText(
        output,
        status,
        (665, 425),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if recording else (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Capture skew: {skew_ms:.1f} ms",
        (665, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return cv2.resize(output, (1152, 648))


cv2.namedWindow("Three Camera Recorder", cv2.WINDOW_NORMAL)

recording = False
writers = []
csv_file = None
csv_writer = None
output_dir = None
frame_count = 0
record_start_ns = None

period_ns = int(1e9 / args.fps)
next_sample_ns = time.monotonic_ns()


def start_recording():
    global recording, writers
    global csv_file, csv_writer, output_dir
    global frame_count, record_start_ns

    name = datetime.now().strftime("take_%Y%m%d_%H%M%S_%f")

    output_dir = Path(args.output) / name
    output_dir.mkdir(parents=True, exist_ok=False)

    writers = []

    for i in range(3):
        path = output_dir / f"camera{i+1}.avi"

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            args.fps,
            (args.width, args.height)
        )

        if not writer.isOpened():
            raise RuntimeError(f"Cannot write {path}")

        writers.append(writer)

    csv_file = (
        output_dir / "frame_timestamps.csv"
    ).open("w", newline="")

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame",
        "sample_time_ns",
        "camera1_time_ns",
        "camera2_time_ns",
        "camera3_time_ns",
        "camera1_sequence",
        "camera2_sequence",
        "camera3_sequence",
        "capture_skew_ms",
        "elapsed_s"
    ])

    metadata = {
        "camera_devices": [
            args.cam1, args.cam2, args.cam3
        ],
        "width": args.width,
        "height": args.height,
        "encoded_fps": args.fps,
        "rotate_cam2_180": args.rotate_cam2,
        "rotate_cam3_180": args.rotate_cam3,
        "timestamp_clock": "time.monotonic_ns",
        "synchronization": (
            "Three latest captured frames sampled together. "
            "Software synchronization, not hardware trigger."
        ),
        "camera1_role": "existing stereo camera 1",
        "camera2_role": "existing stereo camera 2",
        "camera3_role": "close-up MediaPipe finger tracking",
    }

    (output_dir / "recording_meta.json").write_text(
        json.dumps(metadata, indent=2)
    )

    frame_count = 0
    record_start_ns = time.monotonic_ns()
    recording = True

    print("\nRECORDING:", output_dir)


def stop_recording():
    global recording, csv_file

    recording = False

    for writer in writers:
        writer.release()

    if csv_file:
        csv_file.close()
        csv_file = None

    print("\nSAVED:", output_dir)
    print("Frames per camera:", frame_count)


try:
    while True:
        samples = [c.get() for c in cameras]

        if any(s is None for s in samples):
            cv2.waitKey(20)
            continue

        frames = [s[0] for s in samples]
        timestamps = [s[1] for s in samples]
        sequences = [s[2] for s in samples]

        skew_ms = (
            max(timestamps) - min(timestamps)
        ) / 1e6

        preview = make_preview(
            frames,
            recording,
            frame_count,
            skew_ms
        )

        cv2.imshow("Three Camera Recorder", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:
            if recording:
                stop_recording()
            else:
                start_recording()
                next_sample_ns = time.monotonic_ns()

        now = time.monotonic_ns()

        if recording and now >= next_sample_ns:
            if any(
                f.shape[:2] != (args.height, args.width)
                for f in frames
            ):
                raise RuntimeError(
                    "Camera resolution differs from requested size"
                )

            for writer, frame in zip(writers, frames):
                writer.write(frame)

            csv_writer.writerow([
                frame_count,
                now,
                *timestamps,
                *sequences,
                round(skew_ms, 3),
                round(
                    (now - record_start_ns) / 1e9,
                    6
                )
            ])

            frame_count += 1

            next_sample_ns += period_ns

            if next_sample_ns < now:
                next_sample_ns = now + period_ns

finally:
    if recording:
        stop_recording()

    for c in cameras:
        c.close()

    cv2.destroyAllWindows()
