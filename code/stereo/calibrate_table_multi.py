#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--cam1", type=int, default=2)
ap.add_argument("--cam2", type=int, default=0)
ap.add_argument("--id", type=int, default=3)
ap.add_argument("--samples", type=int, default=10)
ap.add_argument("--tag-mm", type=float, default=25.0)
ap.add_argument("--thickness-mm", type=float, default=0.0)
ap.add_argument(
    "--calib",
    type=Path,
    default=BASE / "calibration/results_new/stereo_calibration.npz"
)
ap.add_argument(
    "--out",
    type=Path,
    default=BASE / "calibration/results_new/table_frame_multi.npz"
)
args = ap.parse_args()

with np.load(args.calib) as data:
    K1 = data["K1"].astype(np.float64)
    D1 = data["D1"].astype(np.float64)
    K2 = data["K2"].astype(np.float64)
    D2 = data["D2"].astype(np.float64)
    R = data["R"].astype(np.float64)
    T = data["T"].astype(np.float64).reshape(3, 1)

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, params)

P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = np.hstack([R, T])


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Degenerate geometry")
    return v / n


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{index}")

    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"))
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
        raise RuntimeError(f"Wrong resolution: {size}")
    return cap


def find_tag(frame):
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None

    for c, tag_id in zip(corners, ids.flatten()):
        if int(tag_id) == args.id:
            return np.asarray(c, np.float64).reshape(4, 2)
    return None


def reconstruct(p1, p2):
    u1 = cv2.undistortPoints(
        p1.reshape(-1, 1, 2), K1, D1
    ).reshape(-1, 2)

    u2 = cv2.undistortPoints(
        p2.reshape(-1, 1, 2), K2, D2
    ).reshape(-1, 2)

    h = cv2.triangulatePoints(P1, P2, u1.T, u2.T)

    if np.any(np.abs(h[3]) < 1e-12):
        return None

    xyz = (h[:3] / h[3]).T
    xyz2 = (R @ xyz.T + T).T

    if (not np.isfinite(xyz).all()
            or np.any(xyz[:, 2] <= 0)
            or np.any(xyz2[:, 2] <= 0)):
        return None

    q1, _ = cv2.projectPoints(
        xyz, np.zeros(3), np.zeros(3), K1, D1
    )
    rvec2, _ = cv2.Rodrigues(R)
    q2, _ = cv2.projectPoints(xyz, rvec2, T, K2, D2)

    err1 = np.linalg.norm(q1.reshape(-1, 2) - p1, axis=1)
    err2 = np.linalg.norm(q2.reshape(-1, 2) - p2, axis=1)

    edges = np.linalg.norm(
        xyz - np.roll(xyz, -1, axis=0), axis=1
    ) * 1000

    return xyz, float(max(np.max(err1), np.max(err2))), edges


def fit_plane(points):
    centre = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centre)
    normal = unit(vh[-1])

    # Both cameras are mounted above the tabletop.
    if np.dot(normal, -centre) < 0:
        normal = -normal

    distances = (points - centre) @ normal
    return centre, normal, distances


cap1 = open_camera(args.cam1)
cap2 = open_camera(args.cam2)

samples = []

print("\n===== MULTI-POSITION TABLE CALIBRATION =====")
print("AprilTag:", args.id)
print("Target positions:", args.samples)
print("S = record position")
print("F = fit and save plane")
print("Q = quit")
print("\nKeep cameras stationary. Move only the tag.")

try:
    while True:
        if not (cap1.grab() and cap2.grab()):
            print("Camera grab failed")
            break

        ok1, f1 = cap1.retrieve()
        ok2, f2 = cap2.retrieve()

        if not (ok1 and ok2):
            print("Camera frame retrieval failed")
            break

        c1 = find_tag(f1)
        c2 = find_tag(f2)

        a = f1.copy()
        b = f2.copy()

        for frame, corners in ((a, c1), (b, c2)):
            if corners is not None:
                cv2.polylines(
                    frame,
                    [np.round(corners).astype(np.int32)],
                    True, (0, 255, 0), 2
                )

        left = cv2.resize(a, (640, 360))
        right = cv2.flip(
            cv2.resize(b, (640, 360)), 0
        )  # Preview only

        view = np.hstack([left, right])

        status = (
            f"Samples: {len(samples)}/{args.samples} | "
            f"Cam1 {'OK' if c1 is not None else 'LOST'} | "
            f"Cam2 {'OK' if c2 is not None else 'LOST'} "
            "| S=Capture F=Fit Q=Quit"
        )

        cv2.rectangle(view, (0, 0), (1279, 38),
                      (0, 0, 0), -1)
        cv2.putText(
            view, status, (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 255), 2
        )

        cv2.imshow("Multi-position table calibration", view)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("s"):
            if c1 is None or c2 is None:
                print("NOT SAVED: tag must be visible in both cameras")
                continue

            result = reconstruct(c1, c2)
            if result is None:
                print("NOT SAVED: triangulation failed")
                continue

            xyz, error, edges = result

            if error > 2.0:
                print(f"NOT SAVED: reprojection {error:.2f}px")
                continue

            mean_edge = float(np.mean(edges))
            if abs(mean_edge - args.tag_mm) > 3.0:
                print(
                    f"NOT SAVED: mean tag edge {mean_edge:.2f}mm "
                    f"differs from expected {args.tag_mm:.2f}mm"
                )
                continue

            if np.ptp(edges) > 4.0:
                print("NOT SAVED: tag edge lengths inconsistent")
                continue

            samples.append({
                "points": xyz,
                "reprojection_px": error,
                "edges_mm": edges
            })

            print(
                f"SAVED {len(samples):02d} | "
                f"Reprojection={error:.3f}px | "
                f"Edge mean={mean_edge:.2f}mm | "
                f"Edge range={np.ptp(edges):.2f}mm"
            )

        if key == ord("f"):
            if len(samples) < 6:
                print("Need at least 6 positions")
                continue

            all_points = np.vstack(
                [s["points"] for s in samples]
            )

            centre, normal, residuals = fit_plane(all_points)

            # Verify that sampled positions cover a meaningful area.
            tag_centres = np.array([
                s["points"].mean(axis=0) for s in samples
            ])

            spread = tag_centres - tag_centres.mean(axis=0)
            planar_spread = (
                spread - (spread @ normal)[:, None] * normal
            )
            singular = np.linalg.svd(
                planar_spread, compute_uv=False
            )
            minor_spread_mm = (
                float(singular[1] / np.sqrt(len(samples)) * 1000)
                if len(singular) > 1 else 0.0
            )

            abs_mm = np.abs(residuals) * 1000
            rms_mm = float(
                np.sqrt(np.mean(residuals**2)) * 1000
            )
            p95_mm = float(np.percentile(abs_mm, 95))

            print("\n===== TABLE PLANE FIT =====")
            print("Captured positions:", len(samples))
            print("3D points:", len(all_points))
            print("Plane RMS:", round(rms_mm, 3), "mm")
            print("Plane 95th percentile:",
                  round(p95_mm, 3), "mm")
            print("Minor-axis sample spread:",
                  round(minor_spread_mm, 2), "mm")

            if minor_spread_mm < 40:
                print(
                    "NOT SAVED: positions do not cover "
                    "enough area in two directions."
                )
                continue

            if rms_mm > 3.0 or p95_mm > 6.0:
                print(
                    "NOT SAVED: plane residuals too high. "
                    "Check tag flatness and samples."
                )
                continue

            # Table origin: centre of first sampled tag,
            # projected onto the fitted tabletop.
            first_centre = samples[0]["points"].mean(axis=0)
            origin_tag_plane = (
                first_centre
                - np.dot(first_centre - centre, normal) * normal
            )

            # Use the first tag's X direction, projected
            # into the fitted table plane.
            first = samples[0]["points"]
            x_hint = (
                (first[1] - first[0]) +
                (first[2] - first[3])
            )
            x_axis = unit(
                x_hint - np.dot(x_hint, normal) * normal
            )
            y_axis = unit(np.cross(normal, x_axis))
            z_axis = normal

            R_table_to_cam1 = np.column_stack(
                [x_axis, y_axis, z_axis]
            )

            # Shift origin from tag paper surface to tabletop.
            origin = (
                origin_tag_plane
                - z_axis * (args.thickness_mm / 1000.0)
            )
            t_table_to_cam1 = origin.reshape(3, 1)

            R_cam1_to_table = R_table_to_cam1.T
            t_cam1_to_table = (
                -R_cam1_to_table @ t_table_to_cam1
            )

            camera1_position = (
                -R_cam1_to_table @ t_table_to_cam1
            ).reshape(3)

            camera2_cam1 = (-R.T @ T).reshape(3)
            camera2_position = (
                R_cam1_to_table @
                (camera2_cam1.reshape(3, 1) - t_table_to_cam1)
            ).reshape(3)

            if (camera1_position[2] <= 0
                    or camera2_position[2] <= 0):
                print("NOT SAVED: camera heights not positive")
                continue

            R_table_to_cam2 = R @ R_table_to_cam1
            t_table_to_cam2 = (
                R @ t_table_to_cam1 + T
            )

            # Report how closely each sampled tag centre lies
            # on the fitted plane.
            centre_residual_mm = (
                (tag_centres - centre) @ normal * 1000
            )

            print("Tag centre residuals (mm):")
            print(np.round(centre_residual_mm, 2))

            print("Camera 1 XYZ (mm):",
                  np.round(camera1_position * 1000, 2))
            print("Camera 2 XYZ (mm):",
                  np.round(camera2_position * 1000, 2))

            print(
                "Camera baseline (mm):",
                round(float(
                    np.linalg.norm(
                        camera2_position - camera1_position
                    ) * 1000
                ), 2)
            )

            args.out.parent.mkdir(
                parents=True, exist_ok=True
            )

            if args.out.exists():
                print(
                    "NOT SAVED: output already exists. "
                    "Use a new --out filename to preserve it."
                )
                continue

            np.savez(
                args.out,
                R_table_to_cam1=R_table_to_cam1,
                t_table_to_cam1=t_table_to_cam1,
                R_cam1_to_table=R_cam1_to_table,
                t_cam1_to_table=t_cam1_to_table,
                R_table_to_cam2=R_table_to_cam2,
                t_table_to_cam2=t_table_to_cam2,
                camera1_in_table=camera1_position,
                camera2_in_table=camera2_position,
                table_normal_cam1=normal,
                table_origin_cam1_m=origin,
                plane_rms_mm=rms_mm,
                plane_p95_mm=p95_mm,
                sample_count=len(samples),
                sample_tag_centres_cam1_m=tag_centres,
                sample_corners_cam1_m=all_points,
                sample_centre_residuals_mm=centre_residual_mm,
                tag_id=args.id,
                tag_size_mm=args.tag_mm,
                tag_thickness_mm=args.thickness_mm
            )

            print("\nTABLE FRAME SAVED:")
            print(args.out.resolve())
            break

finally:
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
