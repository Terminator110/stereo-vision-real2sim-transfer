import cv2
import numpy as np
import sys
from pathlib import Path

input_dir = Path("/home/sparx3d/Documents/depth-anything-test-videos")
output_dir = input_dir / "outputs"

filename = Path(sys.argv[1])

# Remove _depth suffix if present
stem = filename.stem

if stem.endswith("_depth"):
    stem = stem[:-6]

# Try original RGB image first
image_path = input_dir / f"{stem}.jpg"

if not image_path.exists():
    image_path = input_dir / f"{stem}.png"

# Fall back to generated depth visualization
if not image_path.exists():
    image_path = output_dir / f"{stem}_depth.png"

# Metric depth NumPy file
depth_path = output_dir / f"{stem}_depth.npy"

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

if not depth_path.exists():
    raise FileNotFoundError(f"Depth file not found: {depth_path}")

image = cv2.imread(str(image_path))
depth = np.load(depth_path)

if image is None:
    raise RuntimeError(f"Unable to read image: {image_path}")

# Match dimensions
if depth.shape != image.shape[:2]:
    depth = cv2.resize(
        depth,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

print("Loaded image:", image_path)
print("Loaded depth:", depth_path)
print("Depth shape:", depth.shape)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        distance = float(depth[y, x])

        print(f"Pixel ({x}, {y}) -> {distance:.3f} meters")

        display = image.copy()

        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

        cv2.putText(
            display,
            f"{distance:.2f} m",
            (max(0, x - 40), max(25, y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Depth Viewer", display)

cv2.namedWindow("Depth Viewer")
cv2.setMouseCallback("Depth Viewer", mouse_callback)

cv2.imshow("Depth Viewer", image)
cv2.waitKey(0)
cv2.destroyAllWindows()