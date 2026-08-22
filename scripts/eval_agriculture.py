"""Application 4: does super-resolution support localised agricultural operations?

Precision agriculture decisions are made per parcel and per plant: which field
to irrigate, where a boundary runs, how many trees stand in a block. This
experiment measures whether those two quantities survive decimation and
reconstruction, using the RSA satellite tiles already used in Chapter 5.

There is no labelled ground truth for these tiles, so the reference is derived
from the high-resolution image itself: a fixed procedure is applied to the HR
tile, and the same procedure is applied unchanged to every reconstruction. What
is measured is therefore agreement with what an analyst would extract from
full-resolution imagery, not accuracy against field survey. That is the right
comparison for the operational question -- whether working from a reconstructed
product leads to the same decision as working from the original.

Two measurements:

  parcel delineation   boundaries from a Felzenszwalb segmentation, scored by
                       precision, recall and F1 against the HR boundaries with
                       a small distance tolerance, plus mean parcel-area error
  canopy counting      individual tree crowns from a Laplacian-of-Gaussian blob
                       detector, scored by count error and by matched-detection
                       F1 within a distance tolerance
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.segmentation import felzenszwalb, find_boundaries

import downstream_common as dc
from eval_satellite import load_pairs
from imresize import imresize

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(THESIS, "results")

BOUNDARY_TOLERANCE = 2        # pixels
CANOPY_TOLERANCE = 6          # pixels
FELZ = dict(scale=200, sigma=0.8, min_size=200)
BLOB = dict(min_sigma=2, max_sigma=6, num_sigma=5, threshold=0.04, overlap=0.4)


def reconstruct(hr01, condition):
    if condition == "HR reference":
        return hr01
    kind, tag = condition.rsplit(" ", 1)
    scale = int(tag.lstrip("x"))
    h, w = hr01.shape[:2]
    lr = imresize(hr01, 1.0 / scale)
    if kind == "Native LR":
        return np.kron(lr, np.ones((scale, scale, 1), dtype=np.float32))
    if kind == "Bicubic":
        return imresize(lr, output_shape=(h, w))
    if kind == "LSMAN":
        return np.clip(np.asarray(dc.lsman(scale)(lr[None], training=False))[0], 0, 1)
    raise ValueError(condition)


# --------------------------------------------------------------------------
# parcel delineation
# --------------------------------------------------------------------------

def parcels(img01):
    seg = felzenszwalb(img01, **FELZ)
    return seg, find_boundaries(seg, mode="thick")


def boundary_scores(ref_b, test_b, tol=BOUNDARY_TOLERANCE):
    """Precision/recall of test boundaries against reference, within ``tol``."""
    if test_b.sum() == 0 or ref_b.sum() == 0:
        return 0.0, 0.0, 0.0
    d_ref = distance_transform_edt(~ref_b)
    d_test = distance_transform_edt(~test_b)
    precision = float((d_ref[test_b] <= tol).mean())
    recall = float((d_test[ref_b] <= tol).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def matched_iou(ref_seg, test_seg):
    """Mean IoU of each reference parcel with its best-overlapping test parcel.

    Segmentation labels carry no meaning across two independent segmentations,
    so parcels must be matched by overlap before their extents can be compared;
    comparing them by label index would be meaningless.
    """
    ref_ids = np.unique(ref_seg)
    if len(ref_ids) == 0:
        return np.nan
    ious = []
    for rid in ref_ids:
        mask = ref_seg == rid
        overlapping, counts = np.unique(test_seg[mask], return_counts=True)
        best = overlapping[np.argmax(counts)]
        inter = counts.max()
        union = mask.sum() + (test_seg == best).sum() - inter
        ious.append(inter / union if union else 0.0)
    return float(np.mean(ious))


# --------------------------------------------------------------------------
# canopy counting
# --------------------------------------------------------------------------

def canopies(img01):
    grey = rgb2gray(img01)
    blobs = blob_log(1.0 - grey, **BLOB)      # crowns are dark against soil
    return blobs[:, :2] if len(blobs) else np.zeros((0, 2))


def match_points(ref, test, tol=CANOPY_TOLERANCE):
    """Optimal one-to-one matching within ``tol``; returns precision/recall/F1."""
    if len(ref) == 0 or len(test) == 0:
        return 0.0, 0.0, 0.0
    cost = np.linalg.norm(ref[:, None, :] - test[None, :, :], axis=2)
    ri, ti = linear_sum_assignment(cost)
    matched = int((cost[ri, ti] <= tol).sum())
    precision = matched / len(test)
    recall = matched / len(ref)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# --------------------------------------------------------------------------

def main():
    pairs = load_pairs(2)                      # HR is the same for any scale
    hr_tiles = [hr for _, _, hr in pairs]
    print(f"{len(hr_tiles)} satellite tiles")

    references = []
    for hr in hr_tiles:
        seg, bnd = parcels(hr)
        references.append((seg, bnd, canopies(hr)))
    n_ref_parcels = np.mean([r[0].max() + 1 for r in references])
    n_ref_canopies = np.mean([len(r[2]) for r in references])
    print(f"  HR reference: {n_ref_parcels:.1f} parcels, "
          f"{n_ref_canopies:.1f} canopies per tile")

    rows = []
    for cond in dc.CONDITIONS:
        b_f1, p_iou, c_f1, c_err = [], [], [], []
        for hr, (ref_seg, ref_bnd, ref_pts) in zip(hr_tiles, references):
            rec = reconstruct(hr, cond)
            seg, bnd = parcels(rec)
            _, _, f1 = boundary_scores(ref_bnd, bnd)
            b_f1.append(f1)
            p_iou.append(matched_iou(ref_seg, seg))

            pts = canopies(rec)
            _, _, cf1 = match_points(ref_pts, pts)
            c_f1.append(cf1)
            c_err.append(abs(len(pts) - len(ref_pts)) / max(len(ref_pts), 1))

        rows.append({
            "condition": cond,
            "boundary_f1": float(np.mean(b_f1)),
            "boundary_f1_std": float(np.std(b_f1)),
            "parcel_iou": float(np.nanmean(p_iou)),
            "canopy_f1": float(np.mean(c_f1)),
            "canopy_f1_std": float(np.std(c_f1)),
            "count_error": float(np.mean(c_err)),
            "n_tiles": len(hr_tiles),
        })
        r = rows[-1]
        print(f"  {cond:<14} boundary F1 {r['boundary_f1']:.3f}  "
              f"parcel IoU {r['parcel_iou']:.3f}  "
              f"canopy F1 {r['canopy_f1']:.3f}  "
              f"count err {r['count_error']*100:5.1f}%")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "results_agriculture.csv")
    df.to_csv(out, index=False)
    print("saved ->", out)


if __name__ == "__main__":
    main()
