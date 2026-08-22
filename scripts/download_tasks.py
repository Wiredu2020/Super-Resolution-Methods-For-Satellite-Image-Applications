"""Fetch the two external task datasets used in Chapter 6.

NWPU VHR-10 \\citep{cheng2014multi} supplies the small-object experiment: 650
annotated images covering ten classes of geospatial object, with one bounding
box per line as ``(x1,y1),(x2,y2),class``.

LEVIR-CD \\citep{chen2020spatial} supplies the change-tracking experiment:
bi-temporal 256x256 patches of the same location with a binary change mask.
Only the validation and test splits are fetched; the training split is an
additional 287 MB and the linear classifier used here does not need it.
"""

import io
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(THESIS, "data")

NWPU_REPO = "SGhazalS/nwpu-vhr10"
NWPU_ROOT = "NWPU VHR-10 dataset"
NWPU_NPZ = os.path.join(DATA, "nwpu_vhr10.npz")

LEVIR_URLS = {
    "val": "https://huggingface.co/datasets/ericyu/LEVIRCD_Cropped_256/resolve/"
           "main/data/val-00000-of-00001-d09d88a7419f2427.parquet",
    "test": "https://huggingface.co/datasets/ericyu/LEVIRCD_Cropped_256/resolve/"
            "main/data/test-00000-of-00001-31d7c3e3444e5b5d.parquet",
}
LEVIR_NPZ = os.path.join(DATA, "levircd.npz")

HEADERS = {"User-Agent": "Mozilla/5.0"}

# NWPU class ids are 1-based in the annotation files.
NWPU_CLASSES = ["airplane", "ship", "storage tank", "baseball diamond",
                "tennis court", "basketball court", "ground track field",
                "harbor", "bridge", "vehicle"]


def fetch(url, timeout=300):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_file(url, path, timeout=900, attempts=6):
    """Stream ``url`` to ``path``, resuming after a truncated transfer.

    The LEVIR-CD parquet files are large enough that a single read is prone to
    an ``IncompleteRead``; this reads in chunks and, on failure, re-requests
    only the missing byte range rather than starting over.
    """
    tmp = path + ".part"
    for attempt in range(1, attempts + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = dict(HEADERS)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                total = r.headers.get("Content-Length")
                total = have + int(total) if total else None
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(tmp, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
            if total is None or os.path.getsize(tmp) >= total:
                os.replace(tmp, path)
                return path
            print(f"    short read ({os.path.getsize(tmp)}/{total}), "
                  f"resuming [{attempt}/{attempts}]")
        except Exception as e:
            print(f"    transfer error [{attempt}/{attempts}]: {str(e)[:60]}")
    raise RuntimeError(f"could not download {url}")


# --------------------------------------------------------------------------
# NWPU VHR-10
# --------------------------------------------------------------------------

def download_nwpu(limit=None):
    if os.path.exists(NWPU_NPZ):
        print("NWPU already cached ->", NWPU_NPZ)
        return

    base = f"https://huggingface.co/datasets/{NWPU_REPO}/resolve/main/"
    ids = [f"{i:03d}" for i in range(1, 651)]
    if limit:
        ids = ids[:limit]

    images, boxes, classes, image_index = [], [], [], []
    kept = 0
    for k, name in enumerate(ids):
        try:
            gt = fetch(base + urllib.parse.quote(
                f"{NWPU_ROOT}/ground truth/{name}.txt")).decode("utf-8", "ignore")
            raw = fetch(base + urllib.parse.quote(
                f"{NWPU_ROOT}/positive image set/{name}.jpg"))
        except Exception as e:
            print(f"  {name}: skipped ({str(e)[:50]})")
            continue

        parsed = []
        for line in gt.splitlines():
            line = line.strip().replace(" ", "")
            if not line:
                continue
            try:
                nums = [int(v) for v in line.replace("(", "").replace(")", "").split(",")]
                if len(nums) == 5:
                    parsed.append(nums)
            except ValueError:
                continue
        if not parsed:
            continue

        im = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        images.append(im)
        for x1, y1, x2, y2, cls in parsed:
            boxes.append([x1, y1, x2, y2])
            classes.append(cls - 1)
            image_index.append(kept)
        kept += 1
        if kept % 50 == 0:
            print(f"  {kept} images, {len(boxes)} objects")

    # Images vary in size, so they are stored as an object array.
    arr = np.empty(len(images), dtype=object)
    for i, im in enumerate(images):
        arr[i] = im
    np.savez_compressed(NWPU_NPZ, images=arr,
                        boxes=np.asarray(boxes, dtype=np.int32),
                        classes=np.asarray(classes, dtype=np.int32),
                        image_index=np.asarray(image_index, dtype=np.int32),
                        class_names=np.asarray(NWPU_CLASSES))
    print(f"NWPU: {kept} images, {len(boxes)} objects -> {NWPU_NPZ}")


# --------------------------------------------------------------------------
# LEVIR-CD
# --------------------------------------------------------------------------

def _decode(cell):
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    return Image.open(io.BytesIO(raw))


def download_levircd():
    if os.path.exists(LEVIR_NPZ):
        print("LEVIR-CD already cached ->", LEVIR_NPZ)
        return

    out = {}
    for split, url in LEVIR_URLS.items():
        path = os.path.join(DATA, f"levircd_{split}.parquet")
        # A failed earlier transfer can leave a truncated or empty file behind,
        # so presence alone is not enough to treat it as cached.
        if os.path.exists(path) and os.path.getsize(path) < 1_000_000:
            os.remove(path)
        if not os.path.exists(path):
            print(f"downloading LEVIR-CD {split} ...")
            download_file(url, path)
        df = pd.read_parquet(path)
        print(f"  {split}: {len(df)} rows, columns {list(df.columns)}")

        a, b, frac = [], [], []
        for _, row in df.iterrows():
            ia = np.asarray(_decode(row["imageA"]).convert("RGB"), dtype=np.uint8)
            ib = np.asarray(_decode(row["imageB"]).convert("RGB"), dtype=np.uint8)
            m = np.asarray(_decode(row["label"]).convert("L"), dtype=np.uint8)
            a.append(ia)
            b.append(ib)
            frac.append(float((m > 127).mean()))
        out[f"{split}_a"] = np.stack(a)
        out[f"{split}_b"] = np.stack(b)
        out[f"{split}_frac"] = np.asarray(frac, dtype=np.float32)

    np.savez_compressed(LEVIR_NPZ, **out)
    for split in LEVIR_URLS:
        f = out[f"{split}_frac"]
        print(f"  {split}: {len(f)} patches, "
              f"{(f > 0.01).mean()*100:.1f}% with >1% changed pixels")
    print("cached ->", LEVIR_NPZ)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    download_levircd()
    download_nwpu()
