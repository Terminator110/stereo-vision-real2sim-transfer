import argparse
import csv
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import pipeline


MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"


def predict_depth(pipe, frame):
    """Return a depth map matching the video resolution."""

    # Convert OpenCV BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to PIL image
    pil_image = Image.fromarray(rgb)

    # Run Depth Anything V2
    result = pipe(pil_image)

    depth = result["predicted_depth"]

    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()

    depth = np.squeeze(depth).astype(np.float32)

    height, width = frame.shape[:2]

    # Resize depth to original video resolution
    if depth.shape != (height, width):
        depth = cv2.resize(
            depth,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    return depth


def track_roi(frame, template, previous_roi, search_radius=80):
    """Locate the original selected patch near its previous position."""

    x, y, w, h = previous_roi
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    left = max(0, x - search_radius)
    top = max(0, y - search_radius)
    right = min(width, x + w + search_radius)
    bottom = min(height, y + h + search_radius)

    search_area = gray[top:bottom, left:right]

    if (
        search_area.shape[0] < template.shape[0]
        or search_area.shape[1] < template.shape[1]
    ):
        return previous_roi, 0.0

    match = cv2.matchTemplate(
        search_area,
        template,
        cv2.TM_CCOEFF_NORMED,
    )

    _, score, _, location = cv2.minMaxLoc(match)

    new_x = left + location[0]
    new_y = top + location[1]

    return (new_x, new_y, w, h), score


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--known-cm", type=float, default=30.0)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Keep ROI at fixed image coordinates",
    )

    args = parser.parse_args()

    if args.every < 1:
        parser.error("--every must be at least 1")

    video_path = Path(args.video)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0
        print("Warning: FPS unavailable; assuming 30 FPS.")

    success, first_frame = cap.read()

    if not success:
        raise RuntimeError("Cannot read first video frame")

    # Select wrist area
    print("Select a small region around the wrist.")
    print("Press ENTER to confirm.")

    roi = cv2.selectROI(
        "Select Wrist",
        first_frame,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi)

    if w == 0 or h == 0:
        raise RuntimeError("No ROI selected")

    roi = (x, y, w, h)

    # Save original patch for visual tracking
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    template = gray[y:y+h, x:x+w].copy()

    print("Loading Depth Anything V2...")

    pipe = pipeline(
        task="depth-estimation",
        model=MODEL_ID,
        device="cpu",
    )

    output_dir = video_path.parent / "wrist_depth_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    baseline = None
    frame_index = 0

    # Process first frame too
    frame = first_frame

    while True:

        if frame_index > 0 and not args.fixed:
            new_roi, score = track_roi(
                frame,
                template,
                roi,
            )

            if score < 0.5 or not np.isfinite(score):
                print(
                    f"Frame {frame_index}: "
                    "Wrist tracking uncertain. Stopping."
                )
            else:
                roi = new_roi

        x, y, w, h = roi

        if frame_index % args.every == 0:

            depth = predict_depth(pipe, frame)

            # Use the center 50% of the selected region
            # to reduce background contamination.
            margin_x = w // 4
            margin_y = h // 4

            x1 = x + margin_x
            x2 = x + w - margin_x
            y1 = y + margin_y
            y2 = y + h - margin_y

            patch = depth[y1:y2, x1:x2]

            valid = patch[np.isfinite(patch) & (patch > 0)]

            if valid.size == 0:
                print(f"Frame {frame_index}: No valid depth")
            else:
                predicted_depth = float(np.median(valid))

                if baseline is None:
                    baseline = predicted_depth

                relative_change = predicted_depth - baseline

                estimated_depth = (
                    args.known_cm / 100.0
                    + relative_change
                )

                timestamp = frame_index / fps

                records.append({
                    "frame": frame_index,
                    "time_sec": timestamp,
                    "predicted_depth_m": predicted_depth,
                    "relative_change_cm": relative_change * 100,
                    "estimated_depth_cm": estimated_depth * 100,
                    "x": x,
                    "y": y,
                })

                print(
                    f"Time: {timestamp:6.2f}s | "
                    f"Raw: {predicted_depth:.3f}m | "
                    f"Change: {relative_change*100:+.2f}cm | "
                    f"Estimate: {estimated_depth*100:.2f}cm"
                )

                # Save annotated sample frame
                display = frame.copy()

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    display,
                    f"Delta: {relative_change*100:+.1f} cm",
                    (max(0, x), max(25, y - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imwrite(
                    str(output_dir / f"frame_{frame_index:06d}.jpg"),
                    display,
                )

        success, frame = cap.read()

        if not success:
            break

        frame_index += 1

    cap.release()

    if not records:
        print("No measurements recorded.")
        return

    # Save measurements
    csv_path = output_dir / "wrist_depth.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    # Plot measured depth changes
    times = [r["time_sec"] for r in records]
    changes = [r["relative_change_cm"] for r in records]

    plt.figure(figsize=(10, 5))
    plt.plot(times, changes, marker="o", markersize=3)
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Estimated relative depth change (cm)")
    plt.title("Wrist Depth Change Over Time")
    plt.grid(True)
    plt.tight_layout()

    graph_path = output_dir / "wrist_depth_graph.png"

    plt.savefig(graph_path, dpi=150)
    plt.close()

    print("\nFinished!")
    print("CSV:", csv_path)
    print("Graph:", graph_path)


if __name__ == "__main__":
    main()