from pathlib import Path
import argparse

import cv2
import mediapipe as mp
import numpy as np


# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "models" / "mediapipe" / "hand_landmarker.task"
)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# HAND CONNECTIONS
# ==========================================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),
]


# ==========================================
# VISUALIZATION
# ==========================================

def draw_hand(frame, landmarks):

    height, width = frame.shape[:2]

    points = []

    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))

    # Draw connections
    for start, end in CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    # Draw landmarks
    for i, point in enumerate(points):

        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1
        )

    return frame


# ==========================================
# PROCESS VIDEO
# ==========================================

def process_video(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError("Could not determine video FPS")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")

    output_video = OUTPUT_DIR / "tracked_hand.mp4"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Cannot create output video")

    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    all_landmarks = []
    all_world_landmarks = []
    timestamps = []
    detected = []
    handedness = []

    frame_idx = 0

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                # Nominal timestamp based on frame index.
                # Suitable for constant-frame-rate videos.
                timestamp_ms = int(round(frame_idx * 1000.0 / fps))

                # RGB conversion
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb)
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                if result.hand_landmarks:

                    hand = result.hand_landmarks[0]

                    xyz = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand],
                        dtype=np.float32
                    )

                    if result.hand_world_landmarks:
                        world = result.hand_world_landmarks[0]

                        world_xyz = np.array(
                            [[lm.x, lm.y, lm.z] for lm in world],
                            dtype=np.float32
                        )
                    else:
                        world_xyz = np.full(
                            (21, 3), np.nan, dtype=np.float32
                        )

                    all_landmarks.append(xyz)
                    all_world_landmarks.append(world_xyz)

                    detected.append(True)

                    label = result.handedness[0][0].category_name
                    handedness.append(label)

                    frame = draw_hand(frame, hand)

                    cv2.putText(
                        frame,
                        f"Hand: {label}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                else:

                    # Preserve frame alignment on tracking failures
                    all_landmarks.append(
                        np.full((21, 3), np.nan, dtype=np.float32)
                    )

                    all_world_landmarks.append(
                        np.full((21, 3), np.nan, dtype=np.float32)
                    )

                    detected.append(False)
                    handedness.append("Unknown")

                timestamps.append(timestamp_ms / 1000.0)

                writer.write(frame)

                frame_idx += 1

                if frame_idx % 30 == 0:
                    print(f"Processed {frame_idx} frames")

    finally:
        cap.release()
        writer.release()

    # ======================================
    # SAVE DATA
    # ======================================

    landmarks_array = np.asarray(all_landmarks, dtype=np.float32)
    world_array = np.asarray(all_world_landmarks, dtype=np.float32)
    detected_array = np.asarray(detected, dtype=bool)

    output_data = OUTPUT_DIR / "hand_landmarks.npz"

    np.savez(
        output_data,
        landmarks=landmarks_array,
        world_landmarks=world_array,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        detected=detected_array,
        handedness=np.asarray(handedness),
        fps=fps
    )

    print("\n=== COMPLETE ===")
    print("Frames:", frame_idx)
    print("Detected:", int(detected_array.sum()))
    print("Missing:", int((~detected_array).sum()))

    if frame_idx:
        print(
            "Detection rate:",
            f"{100 * detected_array.mean():.2f}%"
        )

    print("Saved landmarks:", output_data)
    print("Saved video:", output_video)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video"
    )

    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    process_video(video_path)