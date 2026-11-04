"""Analyze Test 4 without changing its files or requiring MediaPipe inference."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from tracking_quality import load_tracking, save_json, summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'outputs/hand_landmarks.npz')
    parser.add_argument('--video', type=Path, default=ROOT/'videos/vid_1.mp4')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/datetime.now().strftime('test5_analysis_%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    data = load_tracking(args.data)
    report = summarize(data)
    report.update(data=str(args.data.resolve()), video=str(args.video.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    tiles, count = [], 0
    sample_ids = set(np.linspace(0, len(data['detected'])-1, 12, dtype=int).tolist())
    for gap in report['missing_runs']:
        sample_ids.update([gap['start_frame'], gap['end_frame']])
    try:
        with (args.output/'frames.csv').open('w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['frame', 'timestamp_s', 'detected', 'raw_handedness', 'laplacian_variance', 'mean_brightness'])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if count >= len(data['detected']):
                    raise ValueError('Video has more frames than tracking data')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                writer.writerow([count, data['timestamps'][count], bool(data['detected'][count]), data['handedness'][count],
                                 float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean())])
                if count in sample_ids:
                    h, w = frame.shape[:2]
                    tile = np.zeros((260, 360, 3), dtype=np.uint8)
                    scale = min(360/w, 230/h)
                    small = cv2.resize(frame, (round(w*scale), round(h*scale)))
                    tile[:small.shape[0], :small.shape[1]] = small
                    cv2.putText(tile, f'{count}: {"detected" if data["detected"][count] else "missing"}', (8, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    tiles.append(tile)
                count += 1
    finally:
        cap.release()
    if count != len(data['detected']):
        raise ValueError('Video has fewer frames than tracking data')
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)])
        if not cv2.imwrite(str(args.output/'contact_sheet.jpg'), sheet):
            raise RuntimeError('Cannot write contact sheet')
    save_json(args.output/'report.json', report)
    print(f'Detected {report["detected"]}/{report["frames"]} ({report["detection_percent"]:.2f}%)')
    print(f'Within first-to-last detection span: {report["detected_span_percent"]}% (not visibility ground truth)')
    for gap in report['missing_runs']:
        print(f'Missing {gap["start_frame"]}-{gap["end_frame"]}: {gap["duration_s"]:.3f}s; internal={gap["internal"]}')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
