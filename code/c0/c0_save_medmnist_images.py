# c0_save_medmnist_images.py
#
# Download ChestMNIST (224×224) and save each image as a PNG so downstream
# scripts can load them by file path, matching the train/000001 path format
# written by c0_train_medmnist.py.
#
# Output layout:
#   data/medmnist/images/train/000000.png
#   data/medmnist/images/val/000000.png
#   data/medmnist/images/test/000000.png
#
# Run once before C1.

import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from medmnist import ChestMNIST

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
DATA_DIR = DATA_ROOT + "/medmnist"

for split in ["train", "val", "test"]:
    ds = ChestMNIST(split=split, size=224, download=False, root=DATA_DIR, transform=None)
    out_dir = os.path.join(DATA_DIR, "images", split)
    os.makedirs(out_dir, exist_ok=True)

    for idx in tqdm(range(len(ds)), desc=split):
        img, _ = ds[idx]          # PIL Image, grayscale
        img.save(os.path.join(out_dir, f"{idx:06d}.png"))

    print(f"{split}: {len(ds):,} images saved to {out_dir}")
