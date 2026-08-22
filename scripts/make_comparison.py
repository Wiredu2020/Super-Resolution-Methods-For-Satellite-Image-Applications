"""Comparative visual analysis figure.

Reproduces the presentation used in the super-resolution literature and in the
earlier version of this thesis: a row of full reconstructions with the inspected
region marked, and a row of magnified crops beneath, each annotated with its
PSNR and SSIM.  Reading the two rows together shows both global fidelity and the
local detail that the metrics summarise.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from eval_satellite import (bicubic_predict, lanczos_predict, load_pairs,
                            pb_predict, psnr_ssim_y)
from lsman import load_lsman

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(THESIS, "Bright_Wiredu_Nuakoh_Thesis_2023-24", "images")

plt.rcParams.update({"font.family": "serif", "savefig.bbox": "tight"})

INK = "#1b2a33"
ACCENT = "#0e6a6e"
BOX = "#e8b923"


def build(scale=4, index=10, crop=(96, 96, 72)):
    """crop = (row, col, size) in HR pixel coordinates."""
    pairs = load_pairs(scale)
    fname, lr, hr = pairs[index]
    model = load_lsman(scale)

    lr_shown = cv2.resize(lr, (hr.shape[1], hr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    panels = [("Original (HR)", hr, None),
              ("Low resolution", lr_shown, None)]
    for name, fn in (("Bicubic", bicubic_predict),
                     ("Lanczos", lanczos_predict),
                     ("FSRCNN", pb_predict("FSRCNN")),
                     ("ESPCN", pb_predict("ESPCN")),
                     ("LSMAN (proposed)", lambda l, s: np.clip(
                         model(l[None], training=False).numpy()[0], 0, 1))):
        sr = fn(lr, scale)
        panels.append((name, sr, psnr_ssim_y(sr, hr, scale)))

    r, c, k = crop
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(1.85 * n, 4.35),
                             gridspec_kw=dict(hspace=0.04, wspace=0.04))

    for i, (name, img, metrics) in enumerate(panels):
        img = np.clip(img, 0, 1)

        top = axes[0, i]
        top.imshow(img)
        top.add_patch(Rectangle((c, r), k, k, linewidth=1.4,
                                edgecolor=BOX, facecolor="none"))
        proposed = "proposed" in name
        top.set_title(name, fontsize=8.5,
                      color=ACCENT if proposed else INK,
                      weight="bold" if proposed else "normal", pad=4)

        bot = axes[1, i]
        bot.imshow(img[r:r + k, c:c + k])
        for spine in bot.spines.values():
            spine.set_edgecolor(BOX)
            spine.set_linewidth(1.4)
        label = "--" if metrics is None else f"{metrics[0]:.2f} dB / {metrics[1]:.3f}"
        bot.set_xlabel(label, fontsize=8,
                       color=ACCENT if proposed else INK,
                       weight="bold" if proposed else "normal", labelpad=3)

        for ax in (top, bot):
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    out = os.path.join(IMAGES, "lsman_comparative.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out, "| tile:", fname)


if __name__ == "__main__":
    build()
