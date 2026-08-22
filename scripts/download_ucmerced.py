"""Fetch the UC Merced Land Use dataset for the downstream-task experiment.

UC Merced \\citep{yang2010bag} is the standard land-use classification benchmark
for remote sensing: 2100 aerial images of 256x256 pixels at roughly 0.3 m
ground sample distance, evenly divided across 21 land-use classes. The
resolution suits a x4 super-resolution experiment, since decimating to 64x64
leaves a low-resolution image of realistic size.
"""

import io
import os
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(THESIS, "data")
PARQUET = os.path.join(CACHE, "ucmerced.parquet")
NPZ = os.path.join(CACHE, "ucmerced.npz")
URL = ("https://huggingface.co/datasets/blanchon/UC_Merced/resolve/main/"
       "data/train-00000-of-00001.parquet")


def main():
    os.makedirs(CACHE, exist_ok=True)

    if not os.path.exists(PARQUET):
        print("downloading UC Merced ...")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(PARQUET, "wb") as f:
            f.write(r.read())
    print(f"parquet: {os.path.getsize(PARQUET)/1e6:.1f} MB")

    df = pd.read_parquet(PARQUET)
    print("columns:", list(df.columns), "| rows:", len(df))

    img_col = next(c for c in df.columns if "im" in c.lower())
    lab_col = next(c for c in df.columns if "lab" in c.lower())

    images, labels = [], []
    for _, row in df.iterrows():
        cell = row[img_col]
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if im.size != (256, 256):
            im = im.resize((256, 256), Image.BICUBIC)
        images.append(np.asarray(im, dtype=np.uint8))
        labels.append(row[lab_col])

    images = np.stack(images)
    labels = np.asarray(labels)
    print("images:", images.shape, "| classes:", len(np.unique(labels)))

    np.savez_compressed(NPZ, images=images, labels=labels)
    print("cached ->", NPZ)


if __name__ == "__main__":
    main()
