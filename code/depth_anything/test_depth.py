from pathlib import Path
from transformers import pipeline
from PIL import Image
import numpy as np

# Load model once
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    device="cpu"
)

input_dir = Path("/home/sparx3d/Documents/depth-anything-test-videos/")
output_dir = Path("/home/sparx3d/Documents/depth-anything-test-videos/outputs")
output_dir.mkdir(exist_ok=True)

for image_path in sorted(input_dir.iterdir()):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    print(f"Processing {image_path.name}...")

    image = Image.open(image_path).convert("RGB")

    result = pipe(image)

    # Save visualization
    result["depth"].save(
        output_dir / f"{image_path.stem}_depth.png"
    )

    # Save numerical depth
    depth = result["predicted_depth"].detach().cpu().numpy()

    np.save(
        output_dir / f"{image_path.stem}_depth.npy",
        depth
    )

print("All images processed!")