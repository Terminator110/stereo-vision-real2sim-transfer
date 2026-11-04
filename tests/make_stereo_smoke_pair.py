"""Generate synthetic tag-only videos for CLI smoke tests; NOT hand evidence."""
from pathlib import Path
import argparse
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calib', type=Path, required=True)
    parser.add_argument('--table-frame', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.calib, allow_pickle=False) as source:
        cal = {key: source[key] for key in ('K1', 'D1', 'K2', 'D2', 'R', 'T', 'image_size')}
    with np.load(args.table_frame, allow_pickle=False) as source:
        table_r, table_t = source['R_table_to_cam1'], source['t_table_to_cam1'].reshape(3)
    width, height = map(int, cal['image_size'])
    views = [np.full((height, width), 255, dtype=np.uint8) for _ in range(2)]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    square = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=float)*0.0125
    rvec2, _ = cv2.Rodrigues(cal['R'])
    for tag_id, centre in [(4, [0, 0, 0.002]), (5, [0.1, 0.05, 0.002])]:
        points = (square+centre) @ table_r.T+table_t
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 200)
        for index, rvec, tvec in [(0, np.zeros(3), np.zeros(3)), (1, rvec2, cal['T'])]:
            projected, _ = cv2.projectPoints(points, rvec, tvec, cal[f'K{index+1}'], cal[f'D{index+1}'])
            transform = cv2.getPerspectiveTransform(np.array([[0, 0], [199, 0], [199, 199], [0, 199]], np.float32),
                                                    projected.reshape(4, 2).astype(np.float32))
            warped = cv2.warpPerspective(marker, transform, (width, height), flags=cv2.INTER_NEAREST, borderValue=255)
            views[index] = np.minimum(views[index], warped)
    for name, image in zip(['left', 'right'], views):
        writer = cv2.VideoWriter(str(args.output/f'{name}.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (width, height))
        if not writer.isOpened():
            raise RuntimeError('Cannot create synthetic paired-video fixture')
        try:
            for _ in range(8):
                writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()
    print(f'Synthetic tag-only stereo fixture: {args.output}')


if __name__ == '__main__':
    main()
