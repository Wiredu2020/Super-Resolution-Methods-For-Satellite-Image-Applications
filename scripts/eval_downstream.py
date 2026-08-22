"""Application 1: does super-resolution help land-use classification?

Reconstruction metrics measure fidelity to a reference, not usefulness. This
experiment asks a different question: if a land-use classifier is given a
super-resolved image instead of an interpolated one, does it classify better?

UC Merced \\citep{yang2010bag} supplies 2100 aerial images of 256x256 pixels at
roughly 0.3 m ground sample distance across 21 land-use classes. Each image is
treated as the high-resolution reference and decimated by two and by four to
give the observations a coarser sensor would return; the seven conditions of
``downstream_common`` are then compared under one protocol.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import downstream_common as dc

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(THESIS, "data", "ucmerced.npz")
RESULTS = os.path.join(THESIS, "results")
SIDE = 256
TEST_FRACTION = 0.2


def main():
    data = np.load(NPZ)
    hr = data["images"].astype(np.float32) / 255.0
    labels = data["labels"]
    n = len(hr)
    print(f"{n} images, {len(np.unique(labels))} classes")

    lr_by_scale = {}
    for s in dc.SCALES:
        lr_by_scale[s] = dc.decimate(hr, s)
        print(f"  decimated x{s} -> {lr_by_scale[s].shape[1]}px")

    net = dc.backbone(SIDE)
    idx_tr, idx_te = train_test_split(
        np.arange(n), test_size=TEST_FRACTION, random_state=dc.SEED,
        stratify=labels)

    rows = []
    for cond in dc.CONDITIONS:
        make_batch = dc.make_condition(cond, hr, lr_by_scale, SIDE)
        feats = dc.featurise(net, n, make_batch)
        stats, _, _ = dc.fit_eval(feats, labels, idx_tr, idx_te)
        rows.append({"condition": cond, **stats})
        print(f"  {cond:<14} accuracy {stats['accuracy']*100:5.2f}%  "
              f"macro-F1 {stats['macro_f1']:.3f}")
        del feats

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "results_downstream.csv")
    df.to_csv(out, index=False)
    print("saved ->", out)


if __name__ == "__main__":
    main()
