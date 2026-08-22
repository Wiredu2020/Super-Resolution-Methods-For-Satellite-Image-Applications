"""MATLAB-compatible bicubic ``imresize``.

The super-resolution literature generates low-resolution inputs with MATLAB's
``imresize(img, 1/s, 'bicubic')``, which applies an antialiasing filter whose
support widens as the image is shrunk.  OpenCV's ``INTER_CUBIC`` does not
antialias on downscaling, so LR images produced with it are visibly aliased and
pretrained models evaluated on them score roughly 1-2 dB below their published
values.  Reproducing MATLAB's kernel keeps our numbers comparable with published
ones.

Follows Keys (1981), "Cubic convolution interpolation for digital image
processing", with a = -0.5, as MATLAB uses.
"""

import numpy as np


def _cubic(x):
    ax = np.abs(x)
    ax2, ax3 = ax ** 2, ax ** 3
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1) +
            (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2)))


def _contributions(in_len, out_len, scale, kernel_width=4.0):
    if scale < 1:                      # antialias: stretch the kernel
        kernel_width = kernel_width / scale

    x = np.arange(1, out_len + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = int(np.ceil(kernel_width) + 2)

    indices = left[:, None] + np.arange(p)[None, :]
    if scale < 1:
        weights = scale * _cubic(scale * (u[:, None] - indices))
    else:
        weights = _cubic(u[:, None] - indices)

    weights /= np.sum(weights, axis=1, keepdims=True)

    # mirror out-of-range indices back inside the image
    indices = np.minimum(np.maximum(indices - 1, 0), in_len - 1).astype(np.int32)

    # drop columns that ended up all-zero
    keep = np.any(weights, axis=0)
    return weights[:, keep], indices[:, keep]


def _resize_axis(img, weights, indices, axis):
    img = np.swapaxes(img, 0, axis)
    gathered = img[indices]                       # (out, p, ...)
    w = weights.reshape(weights.shape + (1,) * (gathered.ndim - 2))
    out = np.sum(gathered * w, axis=1)
    return np.swapaxes(out, 0, axis)


def imresize(img, scale=None, output_shape=None):
    """Resize ``img`` (float array in [0, 1], HxW or HxWxC) the way MATLAB does."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]

    if output_shape is None:
        out_h, out_w = int(np.ceil(h * scale)), int(np.ceil(w * scale))
        scale_h = scale_w = scale
    else:
        out_h, out_w = output_shape
        scale_h, scale_w = out_h / h, out_w / w

    # resize the axis with the smaller scale first -- fewer operations, and it
    # matches MATLAB's ordering
    order = [0, 1] if scale_h <= scale_w else [1, 0]
    for axis in order:
        if axis == 0:
            weights, indices = _contributions(h, out_h, scale_h)
        else:
            weights, indices = _contributions(w, out_w, scale_w)
        img = _resize_axis(img, weights, indices, axis)

    return np.clip(img, 0.0, 1.0).astype(np.float32)
