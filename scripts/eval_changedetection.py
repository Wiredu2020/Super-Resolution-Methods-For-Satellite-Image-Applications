"""Application 3: does super-resolution help track asset change?

Monitoring built assets means comparing the same location at two dates and
deciding what has changed. LEVIR-CD \\citep{chen2020spatial} supplies exactly
that: bi-temporal 256x256 patches of the same location with a binary mask
marking building change.

Both dates are degraded and reconstructed **identically**, which is the
realistic case -- a coarser sensor affects every acquisition, not one of them.
For each condition the two dates are featurised separately and the descriptor
handed to the classifier is the absolute difference of their features, so the
classifier sees what changed rather than what is present.

Labels are patch-level: a patch counts as changed when more than
``CHANGE_THRESHOLD`` of its pixels are marked in the mask. The classes are
imbalanced, so precision, recall and F1 are reported rather than accuracy
alone.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

import downstream_common as dc

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(THESIS, "data", "levircd.npz")
RESULTS = os.path.join(THESIS, "results")

SIDE = 256
CHANGE_THRESHOLD = 0.01       # fraction of changed pixels for a positive patch
MAX_PATCHES = 600             # per split, to bound CPU time and peak memory


def load_split(data, split):
    # Kept as uint8; a float32 copy of these stacks would run to gigabytes.
    a = data[f"{split}_a"]
    b = data[f"{split}_b"]
    y = (data[f"{split}_frac"] > CHANGE_THRESHOLD).astype(np.int32)
    if len(a) > MAX_PATCHES:
        rng = np.random.default_rng(dc.SEED)
        keep = rng.choice(len(a), MAX_PATCHES, replace=False)
        a, b, y = a[keep], b[keep], y[keep]
    return a, b, y


def descriptor(net, a, b, lr_a, lr_b, cond):
    """|f(date A) - f(date B)| under one condition."""
    fa = dc.featurise(net, len(a), dc.make_condition(cond, a, lr_a, SIDE))
    fb = dc.featurise(net, len(b), dc.make_condition(cond, b, lr_b, SIDE))
    return np.abs(fa - fb)


def main():
    data = np.load(NPZ)
    a_tr, b_tr, y_tr = load_split(data, "val")     # val split fits the classifier
    a_te, b_te, y_te = load_split(data, "test")
    print(f"train {len(a_tr)} patches ({y_tr.mean()*100:.1f}% changed), "
          f"test {len(a_te)} patches ({y_te.mean()*100:.1f}% changed)")

    lr_tr = {s: {} for s in ("a", "b")}
    lr_te = {s: {} for s in ("a", "b")}
    for s in dc.SCALES:
        lr_tr["a"][s], lr_tr["b"][s] = dc.decimate(a_tr, s), dc.decimate(b_tr, s)
        lr_te["a"][s], lr_te["b"][s] = dc.decimate(a_te, s), dc.decimate(b_te, s)
        print(f"  decimated x{s}")

    net = dc.backbone(SIDE)
    n_tr, n_te = len(a_tr), len(a_te)
    idx_tr = np.arange(n_tr)
    idx_te = np.arange(n_tr, n_tr + n_te)

    rows = []
    for cond in dc.CONDITIONS:
        d_tr = descriptor(net, a_tr, b_tr, lr_tr["a"], lr_tr["b"], cond)
        d_te = descriptor(net, a_te, b_te, lr_te["a"], lr_te["b"], cond)
        feats = np.concatenate([d_tr, d_te])
        labels = np.concatenate([y_tr, y_te])

        stats, _, _ = dc.fit_eval(feats, labels, idx_tr, idx_te, binary=True)
        rows.append({"condition": cond, **stats})
        print(f"  {cond:<14} F1 {stats['f1']:.3f}  "
              f"P {stats['precision']:.3f}  R {stats['recall']:.3f}  "
              f"acc {stats['accuracy']*100:5.2f}%", flush=True)
        del d_tr, d_te, feats

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "results_changedetection.csv")
    df.to_csv(out, index=False)
    print("saved ->", out)


if __name__ == "__main__":
    main()
