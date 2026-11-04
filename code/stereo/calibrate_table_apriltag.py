#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--cam1", type=int, default=2)
ap.add_argument("--cam2", type=int, default=0)
ap.add_argument("--id", type=int, default=3)
ap.add_argument("--tag-mm", type=float, default=25.0)
ap.add_argument("--thickness-mm", type=float, default=0.0)
ap.add_argument(
    "--calib",
    type=Path,
    default=ROOT / "calibration/results_new/stereo_calibration.npz"
)
ap.add_argument(
    "--out",
    type=Path,
    default=ROOT / "calibration/results_new/table_frame.npz"
)
args = ap.parse_args()

with np.load(args.calib) as f:
    K1 = f["K1"].astype(np.float64)
    D1 = f["D1"].astype(np.float64)
    K2 = f["K2"].astype(np.float64)
    D2 = f["D2"].astype(np.float64)
    R = f["R"].astype(np.float64)
    T = f["T"].astype(np.float64).reshape(3, 1)

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

detector = cv2.aruco.ArucoDetector(dictionary, params)


def open_cam(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{index}")

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    if size != (1280, 720):
        cap.release()
        raise RuntimeError(f"Unexpected resolution: {size}")

    return cap


def detect_tag(frame):
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None:
        return None

    for c, i in zip(corners, ids.flatten()):
        if int(i) == args.id:
            return np.asarray(
                c, dtype=np.float64
            ).reshape(4, 2)

    return None


def unit(v):
    n = np.linalg.norm(v)

    if n < 1e-10:
        raise ValueError("Degenerate tag geometry")

    return v / n


cap1 = open_cam(args.cam1)
cap2 = open_cam(args.cam2)

print("\n===== APRILTAG TABLE CALIBRATION =====")
print("Tag ID:", args.id)
print("Expected tag edge:", args.tag_mm, "mm")
print("S = save table frame")
print("Q = quit")

try:
    while True:
        if not (cap1.grab() and cap2.grab()):
            print("Camera grab failed")
            break

        ok1, f1 = cap1.retrieve()
        ok2, f2 = cap2.retrieve()

        if not (ok1 and ok2):
            print("Camera retrieve failed")
            break

        p1 = detect_tag(f1)
        p2 = detect_tag(f2)

        display1 = f1.copy()
        display2 = f2.copy()

        for display, pts in ((display1, p1), (display2, p2)):
            if pts is not None:
                poly = np.round(pts).astype(np.int32)
                cv2.polylines(
                    display, [poly], True, (0, 255, 0), 2
                )

                for i, pt in enumerate(poly):
                    cv2.circle(
                        display, tuple(pt), 5, (0, 0, 255), -1
                    )
                    cv2.putText(
                        display, str(i),
                        tuple(pt + np.array([6, -6])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2
                    )

        preview1 = cv2.resize(display1, (640, 360))
        preview2 = cv2.resize(display2, (640, 360))

        # Flip Camera 2 for preview ONLY.
        preview2 = cv2.flip(preview2, 0)

        combined = np.hstack([preview1, preview2])

        status = (
            f"TAG {args.id}: "
            f"C1={'OK' if p1 is not None else 'LOST'} "
            f"C2={'OK' if p2 is not None else 'LOST'} "
            f"| S=SAVE Q=QUIT"
        )

        cv2.rectangle(
            combined, (0, 0), (1250, 36), (0, 0, 0), -1
        )
        cv2.putText(
            combined, status, (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.imshow("Stereo AprilTag table frame", combined)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key != ord("s"):
            continue

        if p1 is None or p2 is None:
            print("NOT SAVED: tag must appear in both cameras")
            continue

        # Undistort both cameras to normalized image coordinates.
        u1 = cv2.undistortPoints(
            p1.reshape(-1, 1, 2), K1, D1
        ).reshape(-1, 2)

        u2 = cv2.undistortPoints(
            p2.reshape(-1, 1, 2), K2, D2
        ).reshape(-1, 2)

        P1 = np.hstack([
            np.eye(3), np.zeros((3, 1))
        ])
        P2 = np.hstack([R, T])

        homogeneous = cv2.triangulatePoints(
            P1, P2, u1.T, u2.T
        )

        if np.any(np.abs(homogeneous[3]) < 1e-10):
            print("NOT SAVED: invalid triangulation")
            continue

        xyz = (
            homogeneous[:3] / homogeneous[3]
        ).T

        xyz2 = (R @ xyz.T + T).T

        if (
            not np.isfinite(xyz).all()
            or np.any(xyz[:, 2] <= 0)
            or np.any(xyz2[:, 2] <= 0)
        ):
            print("NOT SAVED: invalid 3D points")
            continue

        edges = np.linalg.norm(
            xyz - np.roll(xyz, -1, axis=0),
            axis=1
        )

        edge_mm = edges * 1000

        # Build axes from the four reconstructed tag corners.
        # The reference origin is the tag centre.
        origin = xyz.mean(axis=0)

        x_axis = unit(
            (xyz[1] - xyz[0]) +
            (xyz[2] - xyz[3])
        )

        y_hint = (
            (xyz[3] - xyz[0]) +
            (xyz[2] - xyz[1])
        )

        y_axis = unit(
            y_hint - np.dot(y_hint, x_axis) * x_axis
        )

        z_axis = unit(np.cross(x_axis, y_axis))

        # Orient positive table Z toward Camera 1.
        # Cameras are assumed to be above the table.
        camera1_relative = -origin

        if np.dot(z_axis, camera1_relative) < 0:
            y_axis = -y_axis
            z_axis = -z_axis

        R_table_to_cam1 = np.column_stack(
            [x_axis, y_axis, z_axis]
        )

        thickness_m = args.thickness_mm / 1000.0

        # Tag centre is above the tabletop by thickness_m.
        # Move the origin down along the table normal.
        table_origin_cam1 = (
            origin - z_axis * thickness_m
        )

        t_table_to_cam1 = table_origin_cam1.reshape(3, 1)

        R_cam1_to_table = R_table_to_cam1.T
        t_cam1_to_table = (
            -R_cam1_to_table @ t_table_to_cam1
        )

        # Check reprojection in both cameras.
        rvec1 = np.zeros(3)
        tvec1 = np.zeros(3)

        projected1, _ = cv2.projectPoints(
            xyz, rvec1, tvec1, K1, D1
        )

        rvec2, _ = cv2.Rodrigues(R)

        projected2, _ = cv2.projectPoints(
            xyz, rvec2, T, K2, D2
        )

        err1 = np.linalg.norm(
            projected1.reshape(-1, 2) - p1,
            axis=1
        )

        err2 = np.linalg.norm(
            projected2.reshape(-1, 2) - p2,
            axis=1
        )

        cam1_position = (
            -R_table_to_cam1.T @ table_origin_cam1
        )

        cam2_center_cam1 = (-R.T @ T).reshape(3)

        cam2_position = (
            R_table_to_cam1.T @
            (cam2_center_cam1 - table_origin_cam1)
        )

        print("\n===== TAG RECONSTRUCTION =====")
        print("Tag ID:", args.id)
        print("Expected edge:", args.tag_mm, "mm")
        print("Reconstructed edges (mm):",
              np.round(edge_mm, 3))
        print("Mean edge (mm):",
              round(float(np.mean(edge_mm)), 3))
        print("Max Camera 1 reprojection (px):",
              round(float(np.max(err1)), 3))
        print("Max Camera 2 reprojection (px):",
              round(float(np.max(err2)), 3))

        print("\n===== TABLE COORDINATES =====")
        print("Camera 1 XYZ (mm):",
              np.round(cam1_position * 1000, 2))
        print("Camera 2 XYZ (mm):",
              np.round(cam2_position * 1000, 2))

        if (
            abs(float(np.mean(edge_mm)) - args.tag_mm) > 3.0
            or max(float(np.max(err1)), float(np.max(err2))) > 3.0
        ):
            print(
                "\nNOT SAVED: tag dimensions or "
                "reprojection disagreement too large."
            )
            continue

        if cam1_position[2] <= 0 or cam2_position[2] <= 0:
            print("NOT SAVED: camera height is not positive")
            continue

        args.out.parent.mkdir(
            parents=True, exist_ok=True
        )

        np.savez(
            args.out,
            R_table_to_cam1=R_table_to_cam1,
            t_table_to_cam1=t_table_to_cam1,
            R_cam1_to_table=R_cam1_to_table,
            t_cam1_to_table=t_cam1_to_table,
            camera1_in_table=cam1_position,
            camera2_in_table=cam2_position,
            tag_id=args.id,
            tag_size_mm=args.tag_mm,
            board_thickness_mm=args.thickness_mm,
            tag_edges_mm=edge_mm,
            camera1_reprojection_px=err1,
            camera2_reprojection_px=err2,
            tag_corners_cam1_m=xyz
        )

        cv2.imwrite(
            str(args.out.with_name("table_tag_cam1.png")),
            f1
        )
        cv2.imwrite(
            str(args.out.with_name("table_tag_cam2.png")),
            f2
        )

        print("\nTABLE FRAME SAVED:")
        print(args.out.resolve())
        break

finally:
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
