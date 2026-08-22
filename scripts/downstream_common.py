"""Shared harness for the downstream-task experiments.

Every application in Chapter 6 asks the same question of a different task: does
a reconstruction carry the information the task needs? To make the answers
comparable, all of them use one protocol, defined here.

Seven conditions are compared throughout:

  HR reference                the original image, an upper bound
  Native LR   x2, x4          the decimated observation, replicated to full size
  Bicubic     x2, x4          the observation restored by bicubic interpolation
  LSMAN       x2, x4          the observation restored by the proposed network

Features come from a frozen ImageNet-pretrained MobileNetV2
\\citep{sandler2018mobilenetv2}, and the classifier on top is linear. Because
the feature extractor and the classifier are identical across conditions, any
difference in task performance is attributable to the reconstruction alone.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from imresize import imresize
from lsman import load_lsman

SCALES = (2, 4)
BATCH = 8            # small enough that LSMAN's fusion concat stays bounded
SEED = 42

# Order used in every results table.
CONDITIONS = (
    "Native LR x2", "Bicubic x2", "LSMAN x2",
    "Native LR x4", "Bicubic x4", "LSMAN x4",
    "HR reference",
)


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------

_backbone = None


def backbone(side=256):
    """Frozen ImageNet feature extractor, shared across every condition."""
    global _backbone
    if _backbone is None or _backbone.input_shape[1] != side:
        net = tf.keras.applications.MobileNetV2(
            input_shape=(side, side, 3), include_top=False, weights="imagenet",
            pooling="avg")
        net.trainable = False
        _backbone = net
    return _backbone


def featurise(net, n, make_batch, batch=BATCH):
    """Stream batches through ``net``.

    ``make_batch(lo, hi)`` returns images ``lo:hi`` as float32 in [0,1].
    Building each condition lazily keeps peak memory to one batch rather than
    to several full copies of the dataset, which an earlier version exhausted.
    """
    feats = []
    for i in range(0, n, batch):
        arr = np.ascontiguousarray(make_batch(i, min(i + batch, n)), dtype=np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr * 255.0)
        feats.append(np.asarray(net(arr, training=False)))
    return np.concatenate(feats)


# --------------------------------------------------------------------------
# the seven conditions
# --------------------------------------------------------------------------

_models = {}


def lsman(scale):
    if scale not in _models:
        _models[scale] = load_lsman(scale)
    return _models[scale]


def sr_tiled(lr01, scale, tile=128, overlap=8):
    """Super-resolve a large image in overlapping tiles.

    LSMAN concatenates the output of every block before fusing, so peak memory
    grows with ``(D+1) * C`` times the input area -- around a gigabyte for a
    single 800x950 aerial image, which does not fit on a modest machine. The
    network is fully convolutional, so tiling with a small overlap and
    discarding the seams gives essentially the same result at bounded cost.
    """
    model = lsman(scale)
    h, w = lr01.shape[:2]
    out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)

    step = tile - 2 * overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            y0, x0 = max(0, y - overlap), max(0, x - overlap)
            y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
            y0, x0 = max(0, y1 - tile), max(0, x1 - tile)
            patch = lr01[y0:y1, x0:x1]
            pred = np.clip(np.asarray(model(patch[None], training=False))[0], 0, 1)

            # keep only the interior of the tile, except at the image border
            ty0 = 0 if y0 == 0 else overlap
            tx0 = 0 if x0 == 0 else overlap
            ty1 = (y1 - y0) if y1 == h else (y1 - y0) - overlap
            tx1 = (x1 - x0) if x1 == w else (x1 - x0) - overlap
            out[(y0 + ty0) * scale:(y0 + ty1) * scale,
                (x0 + tx0) * scale:(x0 + tx1) * scale] = \
                pred[ty0 * scale:ty1 * scale, tx0 * scale:tx1 * scale]
    return out


def as01(images):
    """Accept uint8 or float stacks and return float32 in [0,1].

    Large image stacks are held as uint8 -- a float32 copy of a few thousand
    256x256 tiles runs to gigabytes -- so conversion happens per batch.
    """
    arr = np.asarray(images)
    return arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8 else arr.astype(np.float32)


def decimate(images, scale, chunk=256):
    """Bicubic decimation, the degradation assumed throughout the thesis.

    Works in chunks so that a uint8 stack is never converted to float in full.
    """
    out = []
    for i in range(0, len(images), chunk):
        out.extend(imresize(im, 1.0 / scale) for im in as01(images[i:i + chunk]))
    return np.stack(out)


def make_condition(condition, hr01, lr_by_scale, out_side):
    """Return a ``make_batch(lo, hi)`` closure for one condition.

    ``hr01`` is the full-resolution stack; ``lr_by_scale[s]`` the decimated
    stack for scale ``s``. Nothing is materialised here -- the closure builds
    each batch on demand.
    """
    if condition == "HR reference":
        return lambda a, b: as01(hr01[a:b])

    kind, tag = condition.rsplit(" ", 1)
    scale = int(tag.lstrip("x"))
    lr = lr_by_scale[scale]

    if kind == "Native LR":
        ones = np.ones((scale, scale, 1), dtype=np.float32)
        return lambda a, b: np.stack([np.kron(im, ones) for im in lr[a:b]])
    if kind == "Bicubic":
        return lambda a, b: np.stack([
            imresize(im, output_shape=(out_side, out_side)) for im in lr[a:b]])
    if kind == "LSMAN":
        model = lsman(scale)
        return lambda a, b: np.clip(
            np.asarray(model(lr[a:b], training=False)), 0, 1)
    raise ValueError(f"unknown condition: {condition}")


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def wilson(p, n, z=1.96):
    """Wilson score interval; reliable for proportions near 0 or 1."""
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return centre - half, centre + half


def fit_eval(feats, labels, idx_tr, idx_te, binary=False):
    """Fit a linear classifier and score it on the held-out split."""
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(feats[idx_tr], labels[idx_tr])
    pred = clf.predict(feats[idx_te])
    truth = labels[idx_te]

    out = {"accuracy": float(accuracy_score(truth, pred)), "n_test": len(idx_te)}
    lo, hi = wilson(out["accuracy"], len(idx_te))
    out["ci_lo"], out["ci_hi"] = float(lo), float(hi)
    if binary:
        out["precision"] = float(precision_score(truth, pred, zero_division=0))
        out["recall"] = float(recall_score(truth, pred, zero_division=0))
        out["f1"] = float(f1_score(truth, pred, zero_division=0))
    else:
        out["macro_f1"] = float(f1_score(truth, pred, average="macro",
                                         zero_division=0))
    return out, pred, truth
