"""WorldStrat real LR/HR pair adapter -- converts WorldStrat's multi-band
GeoTIFFs into the flat HR/LR PNG layout the rest of this pipeline expects,
so training on real (not synthetic-bicubic) degradation is a data-prep
step, not a model or training-loop change.

WorldStrat (Cornebise et al., 2022, https://arxiv.org/abs/2207.06418) pairs
Airbus SPOT 6/7 HR chips (1.5 m/pixel pansharpened, ~1054x1054 px, 12-bit)
with 8-16 temporally-matched Sentinel-2 LR revisits (up to 10 m/pixel,
12-13 bands) per chip. Companion code: https://github.com/worldstrat/worldstrat
Ready-to-use Kaggle mirror: https://www.kaggle.com/datasets/jucor1/worldstrat
(the easiest path if you're already on Kaggle -- "+ Add Input", no download
script needed). License: HR Airbus imagery is CC-BY-NC 4.0 (non-commercial;
fine for thesis use, flag it if this ever goes further); Sentinel-2 imagery
and trained weights are CC BY 4.0.

Why a conversion step at all: `tf.image.decode_image` (used elsewhere in
this pipeline) can't read GeoTIFFs, 12-bit radiometry, or >4-band imagery.
`export_to_png` reads each pair with rasterio, keeps the RGB bands,
percentile-stretches to 8-bit, and writes matched-filename PNGs into
ordinary HR/LR folders -- after that, `data.list_pairs`/`make_dataset`
work completely unchanged.

The on-disk layout differs between the full Zenodo release and the
Kaggle "core" version, so `find_pairs` is glob-pattern-driven rather than
hard-coded. Defaults below are a best guess from the paper/README, NOT
verified against a live mount:

    1. Call inspect(root) first and look at the printed tree.
    2. If hr_glob/lr_glob below don't match what you see, pass your own.
    3. Only then run export_to_png.
"""

import os
from glob import glob

import numpy as np

try:
    import rasterio
except ImportError as e:
    raise ImportError(
        "worldstrat.py needs rasterio to read GeoTIFFs: pip install rasterio"
    ) from e

from PIL import Image


def inspect(root, max_entries=15):
    """Print the first couple of levels under `root` so the actual file
    layout can be confirmed before trusting any glob pattern below."""
    root = str(root)
    print(f"{root}/")
    for entry in sorted(os.listdir(root))[:max_entries]:
        path = os.path.join(root, entry)
        marker = "/" if os.path.isdir(path) else ""
        print(f"  {entry}{marker}")
        if os.path.isdir(path):
            for sub in sorted(os.listdir(path))[:max_entries]:
                print(f"    {sub}")


def find_pairs(root, hr_glob="*/hr.tif", lr_glob="lr_revisit_0.tif"):
    """Discover (hr_path, lr_path) pairs under `root`, one pair per AOI
    subfolder (revisit 0 only -- WorldStrat is natively multi-revisit;
    single-image SR uses one representative revisit per chip).

    Raises loudly instead of silently returning zero/mispaired files, so
    a wrong glob is caught immediately rather than producing a
    quietly-empty or quietly-wrong dataset.
    """
    hr_files = sorted(glob(os.path.join(root, hr_glob)))
    if not hr_files:
        raise FileNotFoundError(
            f"No HR files matched '{hr_glob}' under {root}. "
            "Run worldstrat.inspect(root) and pass the right hr_glob."
        )

    pairs = []
    for hr_path in hr_files:
        aoi_dir = os.path.dirname(hr_path)
        lr_matches = sorted(glob(os.path.join(aoi_dir, lr_glob)))
        if lr_matches:
            pairs.append((hr_path, lr_matches[0]))

    if not pairs:
        raise FileNotFoundError(
            f"Found {len(hr_files)} HR files but none had a matching LR file "
            f"with pattern '{lr_glob}'. Run worldstrat.inspect(root) and pass "
            "the right lr_glob."
        )
    if len(pairs) < len(hr_files):
        print(f"Warning: {len(hr_files) - len(pairs)} HR files had no LR match, skipped.")

    hr_out, lr_out = zip(*pairs)
    return list(hr_out), list(lr_out)


def _read_rgb_8bit(tif_path, band_indices=(1, 2, 3), stretch_percentile=(2, 98)):
    """Read a GeoTIFF, keep 3 bands, percentile-stretch to uint8.

    band_indices are 1-based rasterio band numbers; defaults assume
    band order R,G,B,... (common, but confirm against inspect() output
    or the dataset's band-order documentation -- Sentinel-2 in particular
    is not simply bands 1-3).
    """
    with rasterio.open(tif_path) as src:
        bands = [src.read(b).astype(np.float32) for b in band_indices]
    img = np.stack(bands, axis=-1)

    lo, hi = np.percentile(img, stretch_percentile)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def export_to_png(
    pairs,
    out_hr_dir,
    out_lr_dir,
    hr_band_indices=(1, 2, 3),
    lr_band_indices=(4, 3, 2),
    stretch_percentile=(2, 98),
):
    """Convert (hr_tif, lr_tif) pairs to matched-filename RGB PNGs.

    lr_band_indices defaults to a common Sentinel-2 true-color band
    order (B4/B3/B2 red/green/blue) -- verify against the actual band
    list for whichever Sentinel-2 product (L1C vs L2A) you're using.

    Output filenames are 0.png, 1.png, ... in pairs' order, so HR/LR stay
    trivially matched for data.list_pairs downstream.
    """
    os.makedirs(out_hr_dir, exist_ok=True)
    os.makedirs(out_lr_dir, exist_ok=True)

    for i, (hr_path, lr_path) in enumerate(pairs):
        hr_img = _read_rgb_8bit(hr_path, hr_band_indices, stretch_percentile)
        lr_img = _read_rgb_8bit(lr_path, lr_band_indices, stretch_percentile)

        name = f"{i}.png"
        Image.fromarray(hr_img).save(os.path.join(out_hr_dir, name))
        Image.fromarray(lr_img).save(os.path.join(out_lr_dir, name))

        if (i + 1) % 100 == 0:
            print(f"Exported {i + 1}/{len(pairs)} pairs")

    print(f"Done: {len(pairs)} pairs written to {out_hr_dir} / {out_lr_dir}")
