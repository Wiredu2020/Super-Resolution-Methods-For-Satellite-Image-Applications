"""Application 2: does super-resolution help identify small objects?

Super-resolution should matter most where objects sit close to the sampling
limit, so this experiment measures object recognition as a function of object
size. NWPU VHR-10 \\citep{cheng2014multi} provides 650 annotated images across
ten classes of geospatial object, from aircraft and ships down to individual
vehicles.

Each *whole image* is degraded and reconstructed, so objects are corrupted as
they would be in a real acquisition, and only then are the annotated boxes
cropped out. Each crop is resized to the feature extractor's minimum input of
96x96 -- identically for every condition -- and classified by a linear model.

Objects are grouped by their longest side in the high-resolution reference,
since that is the property the experiment is about. The thresholds are the
terciles of the observed size distribution rather than round numbers, because
the dataset's objects are concentrated between roughly 40 and 140 pixels and a
fixed cut at 32 pixels would leave only a few dozen objects in the smallest
group -- far too few to support a claim. Terciles give each group a comparable
share of the data.

The train/test split is made **by image**, not by object, so that objects from
one scene cannot appear on both sides of the split.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import downstream_common as dc
from imresize import imresize

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(THESIS, "data", "nwpu_vhr10.npz")
RESULTS = os.path.join(THESIS, "results")

CROP = 96                     # MobileNetV2's minimum input
TEST_FRACTION = 0.3
BAND_NAMES = ("small", "medium", "large")

# The full set of 649 images occupies about 1.5 GB once decompressed, which on
# a machine with 8 GB leaves too little headroom and sends the run into swap.
# A random subset keeps several hundred objects in every size band, which is
# ample for the comparison, at a fraction of the memory.
MAX_IMAGES = 320


def reconstruct(img01, condition):
    """Apply one condition to a whole image, returning it at original size."""
    if condition == "HR reference":
        return img01
    kind, tag = condition.rsplit(" ", 1)
    scale = int(tag.lstrip("x"))

    h, w = img01.shape[:2]
    h, w = h - h % scale, w - w % scale
    img01 = img01[:h, :w]
    lr = imresize(img01, 1.0 / scale)

    if kind == "Native LR":
        return np.kron(lr, np.ones((scale, scale, 1), dtype=np.float32))
    if kind == "Bicubic":
        return imresize(lr, output_shape=(h, w))
    if kind == "LSMAN":
        return dc.sr_tiled(lr, scale)
    raise ValueError(condition)


def crops_for(images, boxes, image_index, condition):
    """Every annotated object, cropped from images rebuilt under ``condition``."""
    out = np.empty((len(boxes), CROP, CROP, 3), dtype=np.float32)
    for i in range(len(images)):
        rec = reconstruct(images[i].astype(np.float32) / 255.0, condition)
        h, w = rec.shape[:2]
        for j in np.flatnonzero(image_index == i):
            x1, y1, x2, y2 = boxes[j]
            x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
            y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                out[j] = 0.0
                continue
            patch = rec[y1:y2, x1:x2]
            out[j] = cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    return out


def main():
    data = np.load(NPZ, allow_pickle=True)
    images = data["images"]
    boxes = data["boxes"]
    classes = data["classes"]
    image_index = data["image_index"]
    names = data["class_names"]

    if len(images) > MAX_IMAGES:
        rng = np.random.default_rng(dc.SEED)
        keep = np.sort(rng.choice(len(images), MAX_IMAGES, replace=False))
        remap = -np.ones(len(images), dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        sel = np.flatnonzero(remap[image_index] >= 0)
        images = np.array([images[i] for i in keep], dtype=object)
        boxes, classes = boxes[sel], classes[sel]
        image_index = remap[image_index[sel]]

    print(f"{len(images)} images, {len(boxes)} objects, {len(names)} classes",
          flush=True)

    side = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    cuts = np.percentile(side, [100 / 3, 200 / 3])
    bands = ((BAND_NAMES[0], 0, cuts[0]),
             (BAND_NAMES[1], cuts[0], cuts[1]),
             (BAND_NAMES[2], cuts[1], np.inf))
    band = np.full(len(boxes), BAND_NAMES[2], dtype=object)
    for label, lo, hi in bands:
        band[(side >= lo) & (side < hi)] = label
    for label, lo, hi in bands:
        hi_s = "inf" if np.isinf(hi) else f"{hi:.0f}"
        print(f"  {label:<7} [{lo:.0f}, {hi_s}) px  {(band == label).sum():5d} objects")
    np.save(os.path.join(RESULTS, "smallobject_band_cuts.npy"), cuts)

    img_tr, img_te = train_test_split(
        np.arange(len(images)), test_size=TEST_FRACTION, random_state=dc.SEED)
    idx_tr = np.flatnonzero(np.isin(image_index, img_tr))
    idx_te = np.flatnonzero(np.isin(image_index, img_te))
    print(f"  split by image: {len(idx_tr)} train / {len(idx_te)} test objects")

    net = dc.backbone(CROP)
    rows = []
    for cond in dc.CONDITIONS:
        patches = crops_for(images, boxes, image_index, cond)
        feats = dc.featurise(net, len(patches), lambda a, b: patches[a:b])
        stats, pred, truth = dc.fit_eval(feats, classes, idx_tr, idx_te)
        rows.append({"condition": cond, "band": "all", **stats})

        line = [f"  {cond:<14} all {stats['accuracy']*100:5.2f}%"]
        for label in BAND_NAMES:
            sel = band[idx_te] == label
            if sel.sum() == 0:
                continue
            acc = float((pred[sel] == truth[sel]).mean())
            lo, hi = dc.wilson(acc, int(sel.sum()))
            rows.append({"condition": cond, "band": label, "accuracy": acc,
                         "n_test": int(sel.sum()), "ci_lo": lo, "ci_hi": hi,
                         "macro_f1": np.nan})
            line.append(f"{label} {acc*100:5.2f}%")
        print("   ".join(line), flush=True)
        del patches, feats

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "results_smallobjects.csv")
    df.to_csv(out, index=False)
    print("saved ->", out)


if __name__ == "__main__":
    main()
