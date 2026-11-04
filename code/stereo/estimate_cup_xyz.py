#!/usr/bin/env python3
"""Known-size upright cup fit; callable with recorded masks and explicit calibration."""
from pathlib import Path
import argparse
import cv2
import numpy as np
from scipy.optimize import minimize

BASE = Path(__file__).resolve().parent


def estimate_cup(input_dir, calibration_path, table_path, radius=0.041, height=0.098):
    """Return table-frame base/centre estimates using the existing two-view silhouette fit."""
    OUT = Path(input_dir)
    CAL = Path(calibration_path)
    TABLE = Path(table_path)
    RADIUS = float(radius)
    HEIGHT = float(height)

    SCALE = 0.5

    with np.load(CAL) as f:
        K1 = f["K1"].astype(np.float64)
        D1 = f["D1"].astype(np.float64)
        K2 = f["K2"].astype(np.float64)
        D2 = f["D2"].astype(np.float64)
        R = f["R"].astype(np.float64)
        T = f["T"].reshape(3, 1).astype(np.float64)

    with np.load(TABLE) as f:
        Rtc1 = f["R_table_to_cam1"].astype(np.float64)
        ttc1 = f["t_table_to_cam1"].reshape(3, 1).astype(np.float64)

    R_cam = [
        Rtc1,
        R @ Rtc1
    ]

    t_cam = [
        ttc1,
        R @ ttc1 + T
    ]

    K = [K1, K2]
    D = [D1, D2]

    frames = []
    masks = []

    for i in [1, 2]:
        frame = cv2.imread(str(OUT / f"camera{i}.png"))
        mask = cv2.imread(str(OUT / f"camera{i}_mask.png"), 0)

        if frame is None or mask is None:
            raise SystemExit(f"Missing Camera {i} image or mask")

        if frame.shape[:2] != (720, 1280):
            raise SystemExit("Expected 1280x720 images")

        frames.append(frame)
        masks.append(
            cv2.resize(
                (mask > 127).astype(np.uint8),
                (640, 360),
                interpolation=cv2.INTER_NEAREST
            )
        )

    # Sample points on the known cylinder.
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    heights = [0.0, HEIGHT]

    model = np.array([
        [RADIUS * np.cos(a), RADIUS * np.sin(a), z]
        for z in heights
        for a in angles
    ], dtype=np.float64)


    def project_model(x, y, camera):
        points = model.copy()
        points[:, 0] += x
        points[:, 1] += y

        rvec, _ = cv2.Rodrigues(R_cam[camera])

        projected, _ = cv2.projectPoints(
            points,
            rvec,
            t_cam[camera],
            K[camera],
            D[camera]
        )

        # Reject impossible positions.
        camera_points = (
            R_cam[camera] @ points.T + t_cam[camera]
        ).T

        if np.any(camera_points[:, 2] <= 0):
            return None

        return projected.reshape(-1, 2)


    def render_mask(x, y, camera):
        uv = project_model(x, y, camera)

        if uv is None:
            return None

        uv = uv * SCALE

        hull = cv2.convexHull(
            np.round(uv).astype(np.int32)
        )

        rendered = np.zeros((360, 640), dtype=np.uint8)

        cv2.fillConvexPoly(
            rendered, hull,
            1
        )

        return rendered


    def score_camera(xy, camera):
        rendered = render_mask(xy[0], xy[1], camera)

        if rendered is None:
            return 2.0

        observed = masks[camera]

        area_model = int(rendered.sum())
        area_obs = int(observed.sum())

        if area_model < 20 or area_obs < 20:
            return 2.0

        intersection = np.count_nonzero(
            (rendered > 0) & (observed > 0)
        )

        # F1/Dice silhouette loss.
        # The handle is extra observed area, so the score
        # will not reach 1.0 even for a correct cylinder.
        dice = (
            2.0 * intersection /
            (area_model + area_obs)
        )

        return 1.0 - dice


    def score_both(xy):
        return (
            score_camera(xy, 0) +
            score_camera(xy, 1)
        ) / 2.0


    def initial_xy(camera):
        ys, xs = np.nonzero(masks[camera])

        if len(xs) == 0:
            raise RuntimeError("Empty SAM mask")

        # Approximate the visible object's image centre.
        # This is only an initialization, not the final XYZ.
        uv = np.array(
            [[[np.mean(xs) / SCALE,
               np.mean(ys) / SCALE]]],
            dtype=np.float64
        )

        ray = cv2.undistortPoints(
            uv, K[camera], D[camera]
        ).reshape(2)

        ray_cam = np.array(
            [ray[0], ray[1], 1.0]
        )

        Rtable = R_cam[camera]
        ttable = t_cam[camera].reshape(3)

        camera_centre = -Rtable.T @ ttable
        ray_table = Rtable.T @ ray_cam

        # Intersect ray with plane at cylinder mid-height.
        if abs(ray_table[2]) < 1e-9:
            raise RuntimeError("Ray nearly parallel to table")

        distance = (
            HEIGHT / 2 - camera_centre[2]
        ) / ray_table[2]

        if distance <= 0:
            raise RuntimeError(
                "Initial cup ray points away from workspace"
            )

        position = camera_centre + distance * ray_table

        return position[:2]


    def optimize(start, cameras):
        def objective(xy):
            return np.mean([
                score_camera(xy, c) for c in cameras
            ])

        result = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="Powell",
            bounds=[
                (start[0] - 0.30, start[0] + 0.30),
                (start[1] - 0.30, start[1] + 0.30)
            ],
            options={
                "maxiter": 150,
                "xtol": 1e-5,
                "ftol": 1e-5
            }
        )

        return result.x, float(result.fun)


    print("\n========== CUP XYZ ESTIMATION ==========")

    starts = []

    for cam in [0, 1]:
        xy = initial_xy(cam)
        starts.append(xy)

        print(
            f"Camera {cam + 1} initial XY (mm):",
            np.round(xy * 1000, 2)
        )

    start = np.mean(starts, axis=0)

    # Try several reasonable starting positions.
    candidates = [
        start,
        starts[0],
        starts[1]
    ]

    solutions = []

    for candidate in candidates:
        xy, loss = optimize(candidate, [0, 1])
        solutions.append((loss, xy))

    loss, xy = min(solutions, key=lambda item: item[0])

    # Independent single-view estimates for a diagnostic.
    xy1, loss1 = optimize(starts[0], [0])
    xy2, loss2 = optimize(starts[1], [1])

    disagreement_mm = np.linalg.norm(xy1 - xy2) * 1000

    base = np.array([xy[0], xy[1], 0.0])
    centre = np.array([xy[0], xy[1], HEIGHT / 2])

    print("\n========== RESULTS ==========")

    print("Cup base centre XYZ (mm):")
    print(np.round(base * 1000, 2))

    print("\nCup body centre XYZ (mm):")
    print(np.round(centre * 1000, 2))

    print("\nCylinder silhouette scores:")
    print("Camera 1 Dice:",
          round(1.0 - score_camera(xy, 0), 4))
    print("Camera 2 Dice:",
          round(1.0 - score_camera(xy, 1), 4))

    print("\nIndependent camera XY estimates (mm):")
    print("Camera 1:", np.round(xy1 * 1000, 2))
    print("Camera 2:", np.round(xy2 * 1000, 2))
    print("Difference:", round(disagreement_mm, 2), "mm")

    # Distances to the cup centre.
    for cam in [0, 1]:
        camera_centre = (
            -R_cam[cam].T @ t_cam[cam]
        ).reshape(3)

        distance = np.linalg.norm(
            centre - camera_centre
        )

        print(
            f"Camera {cam + 1} to cup centre:",
            round(distance * 1000, 2), "mm"
        )

    # Generate visual comparison overlays.
    for cam in [0, 1]:
        frame = frames[cam].copy()
        uv = project_model(xy[0], xy[1], cam)

        hull = cv2.convexHull(
            np.round(uv).astype(np.int32)
        )

        cv2.polylines(
            frame, [hull], True,
            (0, 255, 0), 3
        )

        # Project cup body centre.
        rvec, _ = cv2.Rodrigues(R_cam[cam])
        centre_uv, _ = cv2.projectPoints(
            centre.reshape(1, 3),
            rvec,
            t_cam[cam],
            K[cam],
            D[cam]
        )

        centre_pixel = tuple(
            np.round(centre_uv[0, 0]).astype(int)
        )

        cv2.drawMarker(
            frame,
            centre_pixel,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            25,
            3
        )

        cv2.putText(
            frame,
            f"Camera {cam + 1} | Green: cylinder | Red: centre",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        filename = OUT / f"cup_xyz_fit_camera{cam+1}.png"
        cv2.imwrite(str(filename), frame)

    np.savez(
        OUT / "cup_xyz_result.npz",
        base_xyz_table_m=base,
        centre_xyz_table_m=centre,
        camera1_xy_table_m=xy1,
        camera2_xy_table_m=xy2,
        camera_disagreement_mm=disagreement_mm,
        camera1_dice=1.0 - score_camera(xy, 0),
        camera2_dice=1.0 - score_camera(xy, 1)
    )

    print("\nOutputs saved to:", OUT.resolve())

    print("\nIMPORTANT: Inspect both cylinder overlays.")
    print("This is an experimental model fit, not yet")
    print("a validated measurement for robot control.")

    return dict(base_xyz_table_m=base, centre_xyz_table_m=centre,
                camera1_xy_table_m=xy1, camera2_xy_table_m=xy2,
                camera_disagreement_mm=float(disagreement_mm),
                camera1_dice=float(1.0-score_camera(xy, 0)),
                camera2_dice=float(1.0-score_camera(xy, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=BASE/'object_tests')
    parser.add_argument('--calib', type=Path, default=BASE/'calibration/results_new/stereo_calibration.npz')
    parser.add_argument('--table', type=Path, default=BASE/'calibration/results_new/table_frame_multi.npz')
    args = parser.parse_args()
    estimate_cup(args.input_dir, args.calib, args.table)


if __name__ == '__main__':
    main()

