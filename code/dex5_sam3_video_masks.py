#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--frames-dir', required=True, type=Path)
ap.add_argument('--weights', required=True, type=Path)
ap.add_argument('--out', required=True, type=Path)
ap.add_argument('--overlay-every', type=int, default=5)
ap.add_argument('--max-frames', type=int, default=0)
ap.add_argument('--prompt-frame', type=int, default=0)
ap.add_argument('--box', nargs=4, type=int, metavar=('X1','Y1','X2','Y2'),
                help='Previously operator-selected wristband ROI; omit to select interactively')
a = ap.parse_args()

files = sorted(a.frames_dir.glob('*.jpg'))
if a.max_frames:
    files = files[:a.max_frames]
if not files:
    ap.error('No JPG frames found')
if not a.weights.is_file():
    ap.error(f'SAM3 checkpoint not found: {a.weights}')

a.out.mkdir(parents=True, exist_ok=True)
maskdir = a.out / 'masks'
previewdir = a.out / 'sam_overlays'
maskdir.mkdir(exist_ok=True)
previewdir.mkdir(exist_ok=True)

if not 0 <= a.prompt_frame < len(files):
    ap.error(f'--prompt-frame must be 0 to {len(files)-1}')

all_files = files
files = all_files[a.prompt_frame:]

first = cv2.imread(str(files[0]))
print(
    f'Prompt frame: {a.prompt_frame}; '
    f'tracking {len(files)} subsequent frames',
    flush=True
)
h, w = first.shape[:2]

if a.box:
    x,y,x2,y2=a.box;bw,bh=x2-x,y2-y
else:
    x,y,bw,bh=map(int,cv2.selectROI('Select ONLY wristband, press Enter',first,False,False))
    cv2.destroyAllWindows()

if bw < 4 or bh < 4 or x<0 or y<0 or x+bw>w or y+bh>h:
    ap.error('Invalid wristband selection')

box = [x, y, x+bw, y+bh]
(a.out/'sam_prompt.json').write_text(json.dumps({'frame':a.prompt_frame,'box':box},indent=2)+'\n')
print('Tracking ROI:', box, flush=True)

video = a.out / 'sam3_input.mp4'
writer = cv2.VideoWriter(
    str(video),
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (w, h)
)
if not writer.isOpened():
    raise RuntimeError('Cannot create video')

for f in files:
    frame = cv2.imread(str(f))
    if frame is None or frame.shape[:2] != (h, w):
        raise RuntimeError(f'Invalid frame: {f}')
    writer.write(frame)
writer.release()

from ultralytics.models.sam import SAM3VideoPredictor

predictor = SAM3VideoPredictor(overrides={
    'model': str(a.weights.resolve()),
    'task': 'segment',
    'mode': 'predict',
    'imgsz': 640,
    'conf': 0.25,
    'quantize': 16,
    'save': False,
    'verbose': False,
})

results = predictor(
    source=str(video),
    bboxes=[box],
    stream=True
)

prev = ((box[0]+box[2])/2, (box[1]+box[3])/2)
reference_area = None
rows = []

# Frames preceding the prompt have no wristband measurement.
for j, f in enumerate(all_files[:a.prompt_frame]):
    name = f.stem + '.png'
    empty = np.zeros((h, w), dtype='uint8')
    cv2.imwrite(str(maskdir / name), empty)
    rows.append([j, name, 'BEFORE_PROMPT', 0])

for i, result in enumerate(results):
    original_i = i + a.prompt_frame
    if i >= len(files):
        break

    frame = cv2.imread(str(files[i]))
    best = None

    if result.masks is not None:
        for raw in result.masks.data.detach().cpu().numpy():
            if raw.shape != (h, w):
                raw = cv2.resize(
                    raw.astype('float32'),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

            candidate = (raw > 0.5).astype('uint8')
            yy, xx = np.nonzero(candidate)

            if len(xx) < 20:
                continue

            center = (float(xx.mean()), float(yy.mean()))
            dist = np.hypot(
                center[0]-prev[0],
                center[1]-prev[1]
            )

            if best is None or dist < best[0]:
                best = (dist, candidate, center, len(xx))

    mask = np.zeros((h, w), dtype='uint8')
    status = 'MISSING'
    area = 0

    if best is not None:
        dist, candidate, center, area = best

        acceptable = (
            i == 0 or (
                dist <= 80 and (
                    reference_area is None or
                    0.25 * reference_area <= area <= 4 * reference_area
                )
            )
        )

        if acceptable:
            mask = candidate * 255
            status = 'VALID'
            prev = center

            if reference_area is None:
                reference_area = area
        else:
            status = 'REJECTED_JUMP_OR_AREA'

    name = files[i].stem + '.png'
    cv2.imwrite(str(maskdir / name), mask)

    if a.overlay_every > 0 and original_i % a.overlay_every == 0:
        vis = frame.copy()
        vis[mask > 0] = (
            0.4 * vis[mask > 0] +
            0.6 * np.array([0, 255, 0])
        ).astype('uint8')

        cv2.putText(
            vis,
            f'Frame {original_i} {status} pixels={area}',
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.imwrite(str(previewdir / name), vis)

    rows.append([original_i, name, status, int(np.count_nonzero(mask))])

    if i % 10 == 0:
        print(
            f'{original_i}/{len(all_files)} {status} '
            f'mask_pixels={rows[-1][-1]}',
            flush=True
        )

with (a.out / 'sam_tracking.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'mask', 'status', 'mask_pixels'])
    writer.writerows(rows)

print('DONE:', len(rows), '/', len(files), 'frames')
print('Valid:', sum(r[2] == 'VALID' for r in rows))
print('Overlays:', previewdir)
