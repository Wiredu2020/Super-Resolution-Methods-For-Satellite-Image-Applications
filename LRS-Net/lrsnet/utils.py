"""Plotting and reporting helpers. Pure TF/numpy/matplotlib -- no Colab-only
calls -- so this also runs locally later for the CPU-side comparison work."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from .metrics import count_params


def plot_training_history(history, out_path, metric="PSNR"):
    """Loss + metric curves, matching the style of the existing SRCNN/EDSR
    training figures (e.g. images/srcnn_loss, images/EDSR_loss)."""
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(h["loss"], label="train")
    if "val_loss" in h:
        axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if metric in h:
        axes[1].plot(h[metric], label="train")
        if f"val_{metric}" in h:
            axes[1].plot(h[f"val_{metric}"], label="val")
        axes[1].set_title(metric)
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_history_csv(history, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h = history.history
    keys = list(h.keys())
    rows = zip(*[h[k] for k in keys])
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i, row in enumerate(rows):
            writer.writerow([i + 1] + list(row))


def visualize_predictions(model, lr_batch, hr_batch, out_path, n=4):
    """Original / LR / predicted HR grid, same layout as the com_ana /
    msrcnn comparison figures in the thesis."""
    n = min(n, lr_batch.shape[0])
    preds = model.predict(lr_batch[:n])

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        axes[i, 0].imshow(np.clip(lr_batch[i], 0, 1))
        axes[i, 1].imshow(np.clip(preds[i], 0, 1))
        axes[i, 2].imshow(np.clip(hr_batch[i], 0, 1))
        for ax in axes[i]:
            ax.axis("off")
    axes[0, 0].set_title("LR")
    axes[0, 1].set_title("LRS-Net Prediction")
    axes[0, 2].set_title("HR (ground truth)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def model_report(model, name="LRS-Net"):
    """Params row for the efficiency comparison table -- append EDSR/SRGAN/
    SRCNN rows to this locally once weights are back on CPU."""
    params = count_params(model)
    return {
        "model": name,
        "trainable_params": params["trainable"],
        "total_params": params["total"],
    }
