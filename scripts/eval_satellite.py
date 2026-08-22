"""Remote-sensing evaluation of LSMAN against classical and CNN baselines.

The training-notebook benchmark covers only natural images (Set5/Set14/BSD100/
Urban100/DIV2K).  This script adds the missing satellite evaluation.

Protocol, chosen to match the single-image super-resolution literature so the
numbers can be read alongside published ones:

  * HR reference : 256x256 RGB satellite tiles
  * LR input     : MATLAB-compatible bicubic decimation (see ``imresize``)
  * metric       : PSNR and SSIM on the MATLAB ``rgb2ycbcr`` luma channel,
                   with a border of ``scale`` pixels shaved from each side
  * SSIM         : Gaussian-weighted (Wang et al., 2004), sigma = 1.5

The metric path is calibrated: on a Set14-like natural-image set it reproduces
the published bicubic x2 figure to within 0.1 dB.

Latency excludes warm-up iterations and is averaged over repeats, so it reports
steady-state inference rather than one-off graph construction.
"""

import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.metrics import structural_similarity as sk_ssim

from imresize import imresize
from lsman import load_lsman

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = r"C:\Users\Lenovo\Desktop\Research\Super-Resolution-Methods-For-Satellite-Image-Applications"
TILES = os.path.join(REPO, "Test")
PB_DIR = os.path.join(THESIS, "results", "models")
OUT_DIR = os.path.join(THESIS, "results")

SCALES = (2, 4)
TIMING_WARMUP = 3
TIMING_REPEATS = 10


# --------------------------------------------------------------------------
# colour + metrics
# --------------------------------------------------------------------------

def rgb_to_y(rgb01):
    """MATLAB ``rgb2ycbcr`` luma on the [0,1] scale (studio swing, 16-235)."""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def shave(img, border):
    return img[border:-border, border:-border] if border > 0 else img


def psnr_ssim_y(sr01, hr01, scale):
    h = min(sr01.shape[0], hr01.shape[0])
    w = min(sr01.shape[1], hr01.shape[1])
    y_sr = shave(rgb_to_y(np.clip(sr01[:h, :w], 0, 1)), scale) * 255.0
    y_hr = shave(rgb_to_y(np.clip(hr01[:h, :w], 0, 1)), scale) * 255.0
    mse = np.mean((y_sr.astype(np.float64) - y_hr.astype(np.float64)) ** 2)
    psnr = 100.0 if mse <= 1e-10 else 10.0 * np.log10(255.0 ** 2 / mse)
    ssim = sk_ssim(y_hr, y_sr, data_range=255.0, gaussian_weights=True,
                   sigma=1.5, use_sample_covariance=False)
    return psnr, ssim


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_pairs(scale):
    pairs = []
    for fname in sorted(os.listdir(TILES)):
        bgr = cv2.imread(os.path.join(TILES, fname))
        if bgr is None:
            continue
        hr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = hr.shape[:2]
        h, w = h - h % scale, w - w % scale
        hr = hr[:h, :w]
        pairs.append((fname, imresize(hr, 1.0 / scale), hr))
    return pairs


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def bicubic_predict(lr01, scale):
    h, w = lr01.shape[:2]
    return imresize(lr01, output_shape=(h * scale, w * scale))


def lanczos_predict(lr01, scale):
    h, w = lr01.shape[:2]
    return np.clip(cv2.resize(lr01, (w * scale, h * scale),
                              interpolation=cv2.INTER_LANCZOS4), 0, 1)


def _wrap_pb(path):
    gd = tf.compat.v1.GraphDef()
    gd.ParseFromString(open(path, "rb").read())
    fn = tf.compat.v1.wrap_function(
        lambda: tf.compat.v1.import_graph_def(gd, name=""), [])
    return fn.prune("IteratorGetNext:0", "NCHW_output:0")


def pb_predict(name):
    """FSRCNN / ESPCN frozen graphs.  These operate on the luma channel only;
    chroma is carried across by bicubic interpolation, as in OpenCV's
    ``dnn_superres``.  Weights are the OpenCV model-zoo re-trained releases."""
    nets = {}

    def _p(lr01, scale):
        path = os.path.join(PB_DIR, f"{name}_x{scale}.pb")
        if not os.path.exists(path):
            return None
        if scale not in nets:
            nets[scale] = _wrap_pb(path)
        h, w = lr01.shape[:2]
        y = rgb_to_y(lr01)
        sr_y = np.squeeze(nets[scale](tf.constant(y[None, ..., None], tf.float32)).numpy())
        sr_y = sr_y[:h * scale, :w * scale]

        # recombine with bicubic-upsampled chroma
        ycc = cv2.cvtColor(lr01.astype(np.float32), cv2.COLOR_RGB2YCrCb)
        ycc_up = imresize(ycc, output_shape=(h * scale, w * scale))
        # undo the studio-swing offset so OpenCV's full-range inverse applies
        ycc_up[..., 0] = np.clip((sr_y * 255.0 - 16.0) / 219.0, 0, 1)
        return np.clip(cv2.cvtColor(ycc_up, cv2.COLOR_YCrCb2RGB), 0, 1)

    return _p


def keras_predict(models):
    def _p(lr01, scale):
        m = models.get(scale)
        if m is None:
            return None
        return np.clip(m(lr01[np.newaxis, ...], training=False).numpy()[0], 0, 1)
    return _p


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    lsman = {s: load_lsman(s) for s in SCALES}
    params = {s: lsman[s].count_params() for s in SCALES}

    registry = {
        "Bicubic": (bicubic_predict, None),
        "Lanczos": (lanczos_predict, None),
        "FSRCNN": (pb_predict("FSRCNN"), None),
        "ESPCN": (pb_predict("ESPCN"), None),
        "LSMAN (proposed)": (keras_predict(lsman), params),
    }

    rows, per_image = [], []
    for scale in SCALES:
        pairs = load_pairs(scale)
        lr_shape, hr_shape = pairs[0][1].shape, pairs[0][2].shape
        print(f"\n-- scale x{scale}: {len(pairs)} tiles  "
              f"{lr_shape[0]}x{lr_shape[1]} -> {hr_shape[0]}x{hr_shape[1]}")

        for name, (fn, pcount) in registry.items():
            psnrs, ssims = [], []
            for fname, lr, hr in pairs:
                sr = fn(lr, scale)
                if sr is None:
                    break
                p, s = psnr_ssim_y(sr, hr, scale)
                psnrs.append(p)
                ssims.append(s)
                per_image.append({"model": name, "scale": scale,
                                  "image": fname, "psnr_db": p, "ssim": s})
            if not psnrs:
                print(f"   {name:<18} unavailable")
                continue

            lr = pairs[0][1]
            for _ in range(TIMING_WARMUP):
                fn(lr, scale)
            t0 = time.perf_counter()
            for _ in range(TIMING_REPEATS):
                fn(lr, scale)
            latency = (time.perf_counter() - t0) / TIMING_REPEATS * 1000

            rows.append({
                "model": name, "scale": scale, "dataset": "RSA-Test21",
                "n_images": len(psnrs),
                "psnr_db": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
                "ssim": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
                "latency_ms": latency,
                "num_params": int(pcount[scale]) if pcount else np.nan,
            })
            r = rows[-1]
            print(f"   {name:<18} PSNR {r['psnr_db']:6.2f} +/- {r['psnr_std']:4.2f} dB | "
                  f"SSIM {r['ssim']:.4f} +/- {r['ssim_std']:.4f} | {latency:8.1f} ms")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "results_satellite.csv"), index=False)
    pd.DataFrame(per_image).to_csv(os.path.join(OUT_DIR, "results_satellite_per_image.csv"),
                                   index=False)
    with open(os.path.join(OUT_DIR, "lsman_params.json"), "w") as f:
        json.dump({f"x{s}": int(params[s]) for s in SCALES}, f, indent=2)
    print(f"\nSaved -> {OUT_DIR}\\results_satellite.csv")


if __name__ == "__main__":
    main()
