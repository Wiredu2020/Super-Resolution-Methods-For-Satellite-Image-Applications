"""Generate every LSMAN figure used in the thesis, into the LaTeX images/ folder."""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from eval_satellite import (bicubic_predict, lanczos_predict, load_pairs,
                            pb_predict, psnr_ssim_y)
from lsman import load_lsman

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(THESIS, "results")
IMAGES = os.path.join(THESIS, "Bright_Wiredu_Nuakoh_Thesis_2023-24", "images")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})

INK = "#1b2a33"
ACCENT = "#0e6a6e"
WARM = "#b45309"
GREY = "#8а99a1".replace("а", "a")


def save(fig, name):
    path = os.path.join(IMAGES, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("  wrote", name)


# --------------------------------------------------------------------------
# 1. training curves
# --------------------------------------------------------------------------

def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for scale, colour in ((2, ACCENT), (4, WARM)):
        df = pd.read_csv(os.path.join(RESULTS, f"training_history_lsman_x{scale}.csv"))
        axes[0].plot(df.epoch, df.loss, color=colour, lw=1.4, label=f"train, $\\times${scale}")
        axes[0].plot(df.epoch, df.val_loss, color=colour, lw=1.4, ls="--",
                     label=f"validation, $\\times${scale}")
        axes[1].plot(df.epoch, df.keras_psnr, color=colour, lw=1.4, label=f"train, $\\times${scale}")
        axes[1].plot(df.epoch, df.val_keras_psnr, color=colour, lw=1.4, ls="--",
                     label=f"validation, $\\times${scale}")

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("$L_1$ loss")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("PSNR (dB)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "lsman_training.png")


# --------------------------------------------------------------------------
# 2. accuracy vs. model size
# --------------------------------------------------------------------------

def fig_efficiency():
    df = pd.read_csv(os.path.join(RESULTS, "results_summary.csv"))
    df = df[(df.scale == 4) & (df.dataset == "Set5")].dropna(subset=["num_params"]).copy()
    df["label"] = df.model.str.split(" (", regex=False).str[0]
    # published parameter count for the ESRGAN RRDB generator
    df.loc[df.label == "ESRGAN", "num_params"] = 16.7e6

    offsets = {"FSRCNN": (8, 4), "ESPCN": (8, -10), "ESRGAN": (-10, 8),
               "LSMAN": (10, -3)}
    aligns = {"ESRGAN": "right"}

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for _, r in df.iterrows():
        proposed = r.label == "LSMAN"
        ax.scatter(r.num_params / 1e6, r.psnr_db, s=130 if proposed else 55,
                   marker="*" if proposed else "o",
                   color=ACCENT if proposed else INK, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.annotate(r.label, (r.num_params / 1e6, r.psnr_db),
                    textcoords="offset points",
                    xytext=offsets.get(r.label, (8, -3)), fontsize=8,
                    ha=aligns.get(r.label, "left"),
                    color=ACCENT if proposed else INK,
                    weight="bold" if proposed else "normal")
    ax.set_xscale("log")
    ax.set_xlim(3e-3, 3e2)
    ax.set_ylim(df.psnr_db.min() - 0.5, df.psnr_db.max() + 0.6)
    ax.set_xlabel("parameters (millions, log scale)")
    ax.set_ylabel("PSNR (dB) on Set5, $\\times 4$")
    fig.tight_layout()
    save(fig, "lsman_efficiency.png")


# --------------------------------------------------------------------------
# 3. qualitative comparison on a satellite tile
# --------------------------------------------------------------------------

def fig_qualitative(scale=4, index=3):
    pairs = load_pairs(scale)
    fname, lr, hr = pairs[index]
    model = load_lsman(scale)

    panels = [("Low resolution", np.array(cv2.resize(
        lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_NEAREST)), None)]
    for name, fn in (("Bicubic", bicubic_predict),
                     ("Lanczos", lanczos_predict),
                     ("FSRCNN", pb_predict("FSRCNN")),
                     ("LSMAN (proposed)", lambda l, s: np.clip(
                         model(l[None], training=False).numpy()[0], 0, 1))):
        sr = fn(lr, scale)
        panels.append((name, sr, psnr_ssim_y(sr, hr, scale)))
    panels.append(("Ground truth (HR)", hr, None))

    fig, axes = plt.subplots(1, len(panels), figsize=(2.05 * len(panels), 2.45))
    for ax, (name, img, metrics) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1))
        title = name if metrics is None else f"{name}\n{metrics[0]:.2f} dB / {metrics[1]:.3f}"
        ax.set_title(title, fontsize=7.5,
                     color=ACCENT if "proposed" in name else INK,
                     weight="bold" if "proposed" in name else "normal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout()
    save(fig, "lsman_qualitative.png")


# --------------------------------------------------------------------------
# 4. per-image spread on the satellite set
# --------------------------------------------------------------------------

def fig_perimage():
    df = pd.read_csv(os.path.join(RESULTS, "results_satellite_per_image.csv"))
    order = ["Bicubic", "Lanczos", "ESPCN", "FSRCNN", "LSMAN (proposed)"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for ax, scale in zip(axes, (2, 4)):
        sub = df[df.scale == scale]
        data = [sub[sub.model == m].psnr_db.values for m in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color="white", lw=1.3),
                        flierprops=dict(marker="o", ms=2.5, mfc=INK, mec="none"))
        for patch, m in zip(bp["boxes"], order):
            patch.set_facecolor(ACCENT if "proposed" in m else INK)
            patch.set_alpha(1.0 if "proposed" in m else 0.55)
            patch.set_edgecolor("none")
        ax.set_xticklabels([m.split(" (")[0] for m in order], rotation=20, fontsize=8)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"$\\times {scale}$", fontsize=9)
    fig.tight_layout()
    save(fig, "lsman_perimage.png")


# --------------------------------------------------------------------------
# 5. architecture diagram
# --------------------------------------------------------------------------

def _box(ax, x, y, w, h, label, face, text_colour="white", fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=face, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=text_colour, weight="medium")


def _arrow(ax, x0, y0, x1, y1, colour=INK, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=8, lw=0.9, color=colour,
                                 shrinkA=0, shrinkB=0))


def fig_architecture():
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 4.4),
                                 gridspec_kw=dict(height_ratios=[1, 1.15]))

    # ---- top: overall network ----
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    y, h = 0.42, 0.26
    xs = [0.02, 0.14, 0.28, 0.42, 0.56, 0.70, 0.855]
    ws = [0.09, 0.11, 0.11, 0.11, 0.11, 0.12, 0.125]
    labels = ["LR\ninput", "Conv 3$\\times$3\n(shallow $F_0$)", "DAB $\\times$ 1",
              "$\\cdots$", "DAB $\\times$ 6", "Concat + Conv 1$\\times$1\n(fusion)",
              "Sub-pixel $\\uparrow$\n+ Conv 3$\\times$3"]
    faces = [GREY, INK, ACCENT, "#ffffff00", ACCENT, INK, WARM]
    for x, w, lab, fc in zip(xs, ws, labels, faces):
        _box(ax, x, y, w, h, lab, fc, "white" if fc != "#ffffff00" else INK)
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + ws[i], y + h / 2, xs[i + 1], y + h / 2)

    # global residual skip
    _arrow(ax, xs[1] + ws[1] / 2, y, xs[1] + ws[1] / 2, 0.16, colour=WARM, style="-")
    _arrow(ax, xs[1] + ws[1] / 2, 0.16, xs[5] + ws[5] / 2, 0.16, colour=WARM, style="-")
    _arrow(ax, xs[5] + ws[5] / 2, 0.16, xs[5] + ws[5] / 2, y, colour=WARM)
    ax.text((xs[1] + xs[5]) / 2 + 0.05, 0.09, "global residual connection",
            ha="center", fontsize=7, color=WARM, style="italic")

    # dense fusion taps
    for i in (2, 4):
        _arrow(ax, xs[i] + ws[i] / 2, y + h, xs[i] + ws[i] / 2, 0.86, colour=INK, style="-")
        _arrow(ax, xs[i] + ws[i] / 2, 0.86, xs[5] + ws[5] / 2, 0.86, colour=INK, style="-")
    _arrow(ax, xs[5] + ws[5] / 2, 0.86, xs[5] + ws[5] / 2, y + h, colour=INK)
    ax.text((xs[2] + xs[5]) / 2, 0.93, "global feature fusion (all block outputs)",
            ha="center", fontsize=7, color=INK, style="italic")
    ax.set_title("(a)  LSMAN", fontsize=9, loc="left", color=INK)

    # ---- bottom: dual-attention block ----
    bx.set_xlim(0, 1); bx.set_ylim(0, 1); bx.axis("off")
    y2, h2 = 0.42, 0.24
    ws2 = [0.065, 0.125, 0.125, 0.205, 0.215, 0.055]
    gap = 0.028
    xs2, cursor = [], 0.03
    for w in ws2:
        xs2.append(cursor)
        cursor += w + gap
    labels2 = ["$x$", "Conv 3$\\times$3\nReLU", "Conv 3$\\times$3",
               "channel attention\nGAP $\\to$ FC $\\to$ FC $\\to \\sigma$",
               "spatial attention\navg, max $\\to$ Conv 7$\\times$7 $\\to \\sigma$",
               "$\\oplus$"]
    faces2 = [GREY, INK, INK, ACCENT, ACCENT, WARM]
    sizes2 = [8, 7, 7, 6.4, 6.4, 9]
    for x, w, lab, fc, fs in zip(xs2, ws2, labels2, faces2, sizes2):
        _box(bx, x, y2, w, h2, lab, fc, fs=fs)
    for i in range(len(xs2) - 1):
        _arrow(bx, xs2[i] + ws2[i], y2 + h2 / 2, xs2[i + 1], y2 + h2 / 2)

    _arrow(bx, xs2[0] + ws2[0] / 2, y2, xs2[0] + ws2[0] / 2, 0.14, colour=WARM, style="-")
    _arrow(bx, xs2[0] + ws2[0] / 2, 0.14, xs2[5] + ws2[5] / 2, 0.14, colour=WARM, style="-")
    _arrow(bx, xs2[5] + ws2[5] / 2, 0.14, xs2[5] + ws2[5] / 2, y2, colour=WARM)
    bx.text((xs2[0] + xs2[5]) / 2, 0.06, "local residual connection",
            ha="center", fontsize=7, color=WARM, style="italic")
    bx.set_title("(b)  Dual-attention block (DAB)", fontsize=9, loc="left", color=INK)

    fig.tight_layout()
    save(fig, "lsman_architecture.png")


# --------------------------------------------------------------------------
# 6. what the spatial attention gate responds to
# --------------------------------------------------------------------------

def fig_attention(scale=4, index=3):
    import tensorflow as tf
    model = load_lsman(scale)
    pairs = load_pairs(scale)
    _, lr, hr = pairs[index]

    gates = [l for l in model.layers
             if l.__class__.__name__ == "Conv2D" and l.filters == 1]
    probe = tf.keras.Model(model.input, [g.output for g in gates])
    maps = probe(lr[None], training=False)

    show = [0, len(gates) // 2, len(gates) - 1]
    fig, axes = plt.subplots(1, len(show) + 1, figsize=(2.05 * (len(show) + 1), 2.3))
    axes[0].imshow(np.clip(hr, 0, 1))
    axes[0].set_title("HR reference", fontsize=8)
    for ax, i in zip(axes[1:], show):
        m = np.squeeze(maps[i].numpy())
        im = ax.imshow(m, cmap="magma")
        ax.set_title(f"spatial gate, block {i + 1}", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout()
    save(fig, "lsman_attention.png")


if __name__ == "__main__":
    print("writing figures to", IMAGES)
    fig_training_curves()
    fig_efficiency()
    fig_qualitative()
    fig_perimage()
    fig_architecture()
    fig_attention()
