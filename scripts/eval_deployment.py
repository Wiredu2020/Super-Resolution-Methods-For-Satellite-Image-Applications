"""Downlink budget and ground-station deployment measurements.

Two application-level questions are answered here by measurement rather than by
estimate.

The first is the downlink argument: if a spacecraft transmits a scene at
reduced resolution and the ground station reconstructs it, how much data is
saved, and what is given up? Data volumes are measured as encoded bytes rather
than raw pixel counts, since operational downlinks carry compressed products.

The second is deployment: the model is exported to the TensorFlow Lite runtime
used on embedded and edge hardware, and its on-disk size and interpreter
latency are measured directly.
"""

import io
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from eval_satellite import load_pairs
from lsman import load_lsman

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(THESIS, "results")
TFLITE_DIR = os.path.join(THESIS, "results", "tflite")

# Representative small-satellite X-band payload downlink rate.
DOWNLINK_MBPS = 100.0
TIMING_WARMUP = 3
TIMING_REPEATS = 10


def encoded_bytes(img01, fmt="PNG", **kw):
    buf = io.BytesIO()
    Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8)).save(buf, fmt, **kw)
    return buf.getbuffer().nbytes


# --------------------------------------------------------------------------
# downlink budget
# --------------------------------------------------------------------------

def bandwidth_analysis():
    sat = pd.read_csv(os.path.join(RESULTS, "results_satellite.csv"))
    rows = []

    for scale in (1, 2, 4):
        pairs = load_pairs(2)                     # HR is identical for any scale
        hr_png = np.mean([encoded_bytes(hr) for _, _, hr in pairs])
        hr_jpg = np.mean([encoded_bytes(hr, "JPEG", quality=90) for _, _, hr in pairs])

        if scale == 1:
            lr_png, lr_jpg = hr_png, hr_jpg
            psnr = ssim = np.nan
            label = "Full resolution (no reduction)"
        else:
            pairs_s = load_pairs(scale)
            lr_png = np.mean([encoded_bytes(lr) for _, lr, _ in pairs_s])
            lr_jpg = np.mean([encoded_bytes(lr, "JPEG", quality=90) for _, lr, _ in pairs_s])
            r = sat[(sat.scale == scale) & (sat.model == "LSMAN (proposed)")]
            psnr, ssim = float(r.psnr_db.iloc[0]), float(r.ssim.iloc[0])
            label = f"Transmit at 1/{scale}, reconstruct on ground"

        rows.append({
            "strategy": label,
            "scale": scale,
            "png_bytes_per_tile": lr_png,
            "jpeg_bytes_per_tile": lr_jpg,
            "png_reduction_pct": 100 * (1 - lr_png / hr_png),
            "jpeg_reduction_pct": 100 * (1 - lr_jpg / hr_jpg),
            "psnr_db": psnr,
            "ssim": ssim,
        })

    df = pd.DataFrame(rows)

    # Scene-level figures for a 10k x 10k product, scaled from the tile measurements.
    tiles_per_scene = (10000 / 256) ** 2
    df["scene_MB_png"] = df.png_bytes_per_tile * tiles_per_scene / 1e6
    df["downlink_s"] = df.scene_MB_png * 8 / DOWNLINK_MBPS
    df.to_csv(os.path.join(RESULTS, "results_bandwidth.csv"), index=False)

    print("\n=== downlink budget (mean over 21 tiles) ===")
    for _, r in df.iterrows():
        q = "--" if np.isnan(r.psnr_db) else f"{r.psnr_db:.2f} dB / {r.ssim:.4f}"
        print(f"  {r.strategy:<42} PNG {r.png_bytes_per_tile/1024:7.1f} KB "
              f"({r.png_reduction_pct:5.1f}% less)  scene {r.scene_MB_png:7.1f} MB "
              f"= {r.downlink_s:6.1f} s   {q}")
    return df


# --------------------------------------------------------------------------
# embedded export
# --------------------------------------------------------------------------

def export_tflite():
    os.makedirs(TFLITE_DIR, exist_ok=True)
    rows = []

    for scale, side in ((2, 128), (4, 64)):
        # TensorFlow Lite requires a static input shape, so the network is
        # rebuilt at a fixed tile size before conversion. Converting from the
        # Keras model rather than from a concrete function is what causes the
        # trained weights to be embedded as constants in the flatbuffer.
        model = load_lsman(scale, input_shape=(side, side, 3))
        params = model.count_params()

        for label, opts in (("float32", []), ("float16", [tf.lite.Optimize.DEFAULT])):
            conv = tf.lite.TFLiteConverter.from_keras_model(model)
            conv.optimizations = opts
            if label == "float16":
                conv.target_spec.supported_types = [tf.float16]
            try:
                blob = conv.convert()
            except Exception as e:
                print(f"  x{scale} {label}: conversion failed ({str(e)[:70]})")
                continue

            path = os.path.join(TFLITE_DIR, f"lsman_x{scale}_{label}.tflite")
            open(path, "wb").write(blob)

            interp = tf.lite.Interpreter(model_content=blob, num_threads=1)
            interp.allocate_tensors()
            inp = interp.get_input_details()[0]
            out = interp.get_output_details()[0]
            x = np.random.rand(1, side, side, 3).astype(np.float32)

            for _ in range(TIMING_WARMUP):
                interp.set_tensor(inp["index"], x); interp.invoke()
            t0 = time.perf_counter()
            for _ in range(TIMING_REPEATS):
                interp.set_tensor(inp["index"], x); interp.invoke()
            latency = (time.perf_counter() - t0) / TIMING_REPEATS * 1000

            rows.append({
                "scale": scale, "precision": label, "params": int(params),
                "size_kb": len(blob) / 1024, "input": f"{side}x{side}",
                "output": "x".join(str(d) for d in interp.get_tensor(out["index"]).shape[1:3]),
                "latency_ms": latency,
            })
            r = rows[-1]
            print(f"  x{scale} {label:<8} {r['size_kb']:7.1f} KB  "
                  f"{side}x{side} -> {r['output']}  {latency:7.1f} ms")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "results_deployment.csv"), index=False)
    return df


if __name__ == "__main__":
    bw = bandwidth_analysis()
    print("\n=== TensorFlow Lite export ===")
    dep = export_tflite()
    with open(os.path.join(RESULTS, "deployment_summary.json"), "w") as f:
        json.dump({"downlink_mbps": DOWNLINK_MBPS,
                   "tiles_per_10k_scene": (10000 / 256) ** 2}, f, indent=2)
    print("\nsaved -> results_bandwidth.csv, results_deployment.csv")
