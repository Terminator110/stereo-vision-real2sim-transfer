#!/usr/bin/env python3
"""Dual AprilTag stereo wrist tracker. No monocular estimation."""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np


def open_camera(index, size):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    if actual != size:
        cap.release()
        raise RuntimeError(f"Resolution {actual}; expected {size}")

    print(f"Camera /dev/video{index}: {actual}")
    return cap


def detect(frame, detector):
    corners, ids, _ = detector.detectMarkers(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    )

    if ids is None:
        return {}

    return {
        int(i): c.reshape(4, 2).astype(np.float64)
        for c, i in zip(corners, ids.flatten())
    }


def triangulate(a, b, cal):
    a_norm = cv2.undistortPoints(
        a.reshape(-1, 1, 2), cal["K1"], cal["D1"]
    ).reshape(-1, 2)

    b_norm = cv2.undistortPoints(
        b.reshape(-1, 1, 2), cal["K2"], cal["D2"]
    ).reshape(-1, 2)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((cal["R"], cal["T"]))

    homogeneous = cv2.triangulatePoints(
        P1, P2, a_norm.T, b_norm.T
    )

    return (homogeneous[:3] / homogeneous[3:4]).T


def reprojection(pts, a, b, cal):
    rvec2, _ = cv2.Rodrigues(cal["R"])

    q1, _ = cv2.projectPoints(
        pts, np.zeros(3), np.zeros(3),
        cal["K1"], cal["D1"]
    )

    q2, _ = cv2.projectPoints(
        pts, rvec2, cal["T"],
        cal["K2"], cal["D2"]
    )

    errors = np.concatenate([
        np.linalg.norm(q1.reshape(-1, 2) - a, axis=1),
        np.linalg.norm(q2.reshape(-1, 2) - b, axis=1)
    ])

    return float(np.mean(errors))


def reconstruct(tag_id, a, b, cal, tag_mm, max_error):
    try:
        pts = triangulate(a, b, cal)

        if not np.isfinite(pts).all():
            return None

        z2 = (cal["R"] @ pts.T + cal["T"])[2]

        if np.any(pts[:, 2] <= 0) or np.any(z2 <= 0):
            return None

        edge_lengths = np.linalg.norm(
            pts - np.roll(pts, -1, axis=0), axis=1
        )

        edge_mm = float(np.mean(edge_lengths) * 1000)
        error = reprojection(pts, a, b, cal)

        if error > max_error:
            return None

        if abs(edge_mm - tag_mm) > 0.15 * tag_mm:
            return None

        return {
            "id": tag_id,
            "xyz": pts.mean(axis=0),
            "error": error,
            "edge_mm": edge_mm
        }

    except (cv2.error, ValueError, FloatingPointError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=int, default=2)
    parser.add_argument("--right", type=int, default=0)
    parser.add_argument("--tag-mm", type=float, default=25)
    parser.add_argument("--ids", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--max-reproj", type=float, default=3.0)
    parser.add_argument(
        "--calib",
        default="stereo/calibration/results/stereo_calibration.npz"
    )
    parser.add_argument(
        "--output",
        default="stereo/wrist_trajectory.csv"
    )
    args = parser.parse_args()

    if args.tag_mm <= 0:
        parser.error("--tag-mm must be positive")

    with np.load(args.calib) as data:
        cal = {
            k: data[k]
            for k in ("K1", "D1", "K2", "D2", "R", "T")
        }
        size = tuple(map(int, data["image_size"]))

    cal["T"] = cal["T"].reshape(3, 1)

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    detector = cv2.aruco.ArucoDetector(dictionary)

    cam1 = open_camera(args.left, size)
    cam2 = open_camera(args.right, size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "time_s", "marker_id", "valid",
        "x_cam1_m", "y_cam1_m", "z_cam1_m",
        "mean_reprojection_px", "mean_tag_edge_mm",
        "source", "shared_ids", "valid_ids",
        "tag5_x_m", "tag5_y_m", "tag5_z_m",
        "tag6_x_m", "tag6_y_m", "tag6_z_m",
        "tag5_error_px", "tag6_error_px",
        "tag_center_separation_mm"
    ]

    rows = []
    recording = False
    start = time.monotonic()

    print("\nDual-tag stereo tracking")
    print("IDs:", args.ids)
    print("R = record/stop | Q = quit")
    print("No monocular tracking or prediction\n")

    try:
        while True:
            ok1 = cam1.grab()
            ok2 = cam2.grab()

            if not (ok1 and ok2):
                print("Camera grab failed")
                break

            ok1, frame1 = cam1.retrieve()
            ok2, frame2 = cam2.retrieve()

            if not (ok1 and ok2):
                print("Camera retrieval failed")
                break

            timestamp = time.monotonic() - start

            detected1 = detect(frame1, detector)
            detected2 = detect(frame2, detector)

            common = sorted(
                set(detected1) &
                set(detected2) &
                set(args.ids)
            )

            measurements = {}

            for tag_id in common:
                measurement = reconstruct(
                    tag_id,
                    detected1[tag_id],
                    detected2[tag_id],
                    cal,
                    args.tag_mm,
                    args.max_reproj
                )

                if measurement is not None:
                    measurements[tag_id] = measurement

            # Select the valid shared tag with the lowest
            # stereo reprojection error.
            if measurements:
                best = min(
                    measurements.values(),
                    key=lambda m: m["error"]
                )
                selected_id = best["id"]
                xyz = best["xyz"]
                error = best["error"]
                edge_mm = best["edge_mm"]
                valid = 1
                source = f"STEREO_TAG_{selected_id}"
            else:
                selected_id = None
                xyz = np.full(3, np.nan)
                error = np.nan
                edge_mm = np.nan
                valid = 0
                source = "LOST"

            def tag_xyz(tag_id):
                if tag_id not in measurements:
                    return [np.nan] * 3
                return measurements[tag_id]["xyz"].tolist()

            def tag_error(tag_id):
                if tag_id not in measurements:
                    return np.nan
                return measurements[tag_id]["error"]

            if 5 in measurements and 6 in measurements:
                separation = float(
                    np.linalg.norm(
                        measurements[5]["xyz"] -
                        measurements[6]["xyz"]
                    ) * 1000
                )
            else:
                separation = np.nan

            if recording:
                rows.append([
                    timestamp,
                    selected_id if selected_id is not None else "",
                    valid,
                    *xyz.tolist(),
                    error,
                    edge_mm,
                    source,
                    ",".join(map(str, common)),
                    ",".join(map(str, sorted(measurements))),
                    *tag_xyz(5),
                    *tag_xyz(6),
                    tag_error(5),
                    tag_error(6),
                    separation
                ])

            # Draw on copies. Never flip the images used
            # for triangulation.
            for frame, detected in (
                (frame1, detected1),
                (frame2, detected2)
            ):
                for tag_id, corners in detected.items():
                    polygon = np.round(corners).astype(np.int32)

                    color = (
                        (0, 255, 0)
                        if tag_id in measurements
                        else (0, 165, 255)
                    )

                    cv2.polylines(
                        frame, [polygon], True, color, 2
                    )

                    cv2.putText(
                        frame,
                        f"ID {tag_id}",
                        tuple(polygon[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2
                    )

            left_view = cv2.resize(frame1, (640, 360))

            # Vertical flip is for Camera 2 DISPLAY ONLY.
            right_view = cv2.resize(
                cv2.flip(frame2, 0), (640, 360)
            )

            preview = cv2.hconcat([left_view, right_view])

            cv2.rectangle(
                preview, (0, 0), (1280, 80),
                (0, 0, 0), -1
            )

            lines = [
                f"{source} | shared={common} | recording={recording}",
                (
                    f"XYZ camera1: "
                    f"{xyz[0]*1000:.1f}, "
                    f"{xyz[1]*1000:.1f}, "
                    f"{xyz[2]*1000:.1f} mm"
                ),
                (
                    f"Reproj={error:.2f}px | "
                    f"edge={edge_mm:.1f}mm | "
                    f"tag separation={separation:.1f}mm"
                )
            ]

            for i, line in enumerate(lines):
                cv2.putText(
                    preview, line,
                    (10, 23 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 1
                )

            cv2.imshow("Dual AprilTag Stereo Wrist", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                recording = not recording
                print("Recording:", "ON" if recording else "OFF")

            elif key == ord("q"):
                break

    finally:
        cam1.release()
        cam2.release()
        cv2.destroyAllWindows()

        if rows:
            with output.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(rows)

            print("Saved:", output)
            print("Recorded frames:", len(rows))


if __name__ == "__main__":
    main()
