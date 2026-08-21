import os
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import color

# ==========================================================
# Paths
# ==========================================================
INPUT_DIR = "input"
OUTPUT_DIR = "output"

# ==========================================================
# Color spaces
# ==========================================================
color_spaces = [
    "RGB",
    "HSV",
    "HLS",
    "CIELAB",
    "LUV",
    "XYZ",
    "YCbCr",
    "HED",
    "H",
    "E",
    "OD",
]

for c in color_spaces:
    os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

# ==========================================================
# Image list
# ==========================================================
extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

files = []
for ext in extensions:
    files.extend(Path(INPUT_DIR).glob(f"*{ext}"))
    files.extend(Path(INPUT_DIR).glob(f"*{ext.upper()}"))

files = sorted(files)

print(f"Found {len(files)} images.")

if len(files) == 0:
    raise RuntimeError("No images found. Check INPUT_DIR.")

# Random image to visualize each run
example_idx = random.randint(0, len(files) - 1)
print(f"Example image: {files[example_idx].name}")

# ==========================================================
# Helper
# ==========================================================
def normalize_for_display(img):
    img = np.nan_to_num(img)
    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    return img


# ==========================================================
# Process images
# ==========================================================
for idx, file in enumerate(tqdm(files)):

    bgr = cv2.imread(str(file))

    if bgr is None:
        print(f"Skipping unreadable file: {file}")
        continue

    name = file.stem

    # ======================================================
    # RGB
    # ======================================================
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_float = rgb.astype(np.float32) / 255.0

    # ======================================================
    # General color spaces
    # ======================================================
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luv = cv2.cvtColor(rgb, cv2.COLOR_RGB2Luv)
    xyz = cv2.cvtColor(rgb, cv2.COLOR_RGB2XYZ)

    # OpenCV gives YCrCb, so reorder to standard YCbCr
    ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycbcr = ycbcr[:, :, [0, 2, 1]]

    # ======================================================
    # Histology-specific spaces
    # ======================================================
    hed = color.rgb2hed(rgb_float).astype(np.float32)

    H = hed[:, :, 0]
    E = hed[:, :, 1]

    od = -np.log((rgb.astype(np.float32) + 1.0) / 256.0)

    # ======================================================
    # Save raw arrays only
    # ======================================================
    np.save(f"{OUTPUT_DIR}/RGB/{name}.npy", rgb)
    np.save(f"{OUTPUT_DIR}/HSV/{name}.npy", hsv)
    np.save(f"{OUTPUT_DIR}/HLS/{name}.npy", hls)
    np.save(f"{OUTPUT_DIR}/CIELAB/{name}.npy", lab)
    np.save(f"{OUTPUT_DIR}/LUV/{name}.npy", luv)
    np.save(f"{OUTPUT_DIR}/XYZ/{name}.npy", xyz)
    np.save(f"{OUTPUT_DIR}/YCbCr/{name}.npy", ycbcr)
    np.save(f"{OUTPUT_DIR}/HED/{name}.npy", hed)
    np.save(f"{OUTPUT_DIR}/H/{name}.npy", H)
    np.save(f"{OUTPUT_DIR}/E/{name}.npy", E)
    np.save(f"{OUTPUT_DIR}/OD/{name}.npy", od)

    # ======================================================
    # Show one random example only
    # ======================================================
    if idx == example_idx:

        fig, ax = plt.subplots(3, 4, figsize=(18, 12))

        ax[0, 0].imshow(rgb)
        ax[0, 0].set_title("RGB")

        ax[0, 1].imshow(normalize_for_display(hsv))
        ax[0, 1].set_title("HSV channels")

        ax[0, 2].imshow(normalize_for_display(hls))
        ax[0, 2].set_title("HLS channels")

        ax[0, 3].imshow(normalize_for_display(lab))
        ax[0, 3].set_title("CIELAB channels")

        ax[1, 0].imshow(normalize_for_display(luv))
        ax[1, 0].set_title("LUV channels")

        ax[1, 1].imshow(normalize_for_display(xyz))
        ax[1, 1].set_title("XYZ channels")

        ax[1, 2].imshow(normalize_for_display(ycbcr))
        ax[1, 2].set_title("YCbCr channels")

        ax[1, 3].imshow(normalize_for_display(hed))
        ax[1, 3].set_title("HED channels")

        ax[2, 0].imshow(normalize_for_display(H), cmap="gray")
        ax[2, 0].set_title("Hematoxylin")

        ax[2, 1].imshow(normalize_for_display(E), cmap="gray")
        ax[2, 1].set_title("Eosin")

        ax[2, 2].imshow(normalize_for_display(od.mean(axis=2)), cmap="gray")
        ax[2, 2].set_title("OD mean")

        ax[2, 3].axis("off")

        for a in ax.ravel():
            a.axis("off")

        plt.tight_layout()
        plt.show()

print("Finished.")
