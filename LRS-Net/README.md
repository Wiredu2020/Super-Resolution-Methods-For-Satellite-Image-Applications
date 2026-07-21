# LRS-Net — Colab/Kaggle Training

GPU training for LRS-Net (Lightweight Remote-Sensing Super-Resolution Network). CPU-only work — model comparison, the Pareto plot, downstream-task evaluation — happens locally afterward using the outputs this produces.

## Structure

```
Colab/
  LRSNet_Training.ipynb   Run top to bottom on Colab or Kaggle
  lrsnet/                 Model, data pipeline, losses, metrics, plotting
  requirements.txt
```

`lrsnet/` has no Colab-specific calls, so it also imports locally later for CPU-side work.

## Data

Expected layout, uploaded to Drive (or a Kaggle dataset):

```
DATA_DIR/
  HR/train/*.png   HR/val/*.png
  LR/train/*.png   LR/val/*.png   # omit to use synthetic bicubic degradation instead
```

If using real paired data, filenames in `LR/<split>` and `HR/<split>` must correspond 1:1 in sorted order.

### Real LR/HR pairs (vs. the synthetic bicubic downsampling used elsewhere in this thesis)

| Dataset | Pairing | Resolution | Notes |
|---|---|---|---|
| [WorldStrat](https://arxiv.org/abs/2207.06418) | Real: Airbus SPOT 6/7 (HR) + temporally-matched Sentinel-2 (LR) | 1.5 m / 10 m | ~10,000 km², 3,504 areas of interest, built specifically for satellite SR benchmarking |
| [SEN2VENµS](https://www.nature.com/articles/s41597-023-02538-9) | Real: Sentinel-2 + VENµS, same-day acquisitions | 5 m / 10-20 m | 132,955 patches, 29 locations, very clean pairing (≤30 min apart) |
| [OLI2MSI](https://github.com/wjwjww/OLI2MSI) | Real: Landsat-8 OLI (LR) + Sentinel-2 MSI (HR) | 30 m / 10 m | Smallest/simplest to get started with, fixed x3 scale |

Any of these swaps in via `LR_TRAIN_DIR`/`LR_VAL_DIR` in the notebook config cell — no model or training-loop changes needed.

## Outputs

Written to `OUTPUT_DIR` on Drive:

```
weights/lrsnet_best.weights.h5
weights/lrsnet_final.weights.h5
figures/lrsnet_loss_psnr.png
figures/lrsnet_val_predictions.png
history.csv
model_report.json
```

Copy this folder into `Models/LRSNet/` locally when done, matching the existing `Models/SRCNN/`, `Models/EDSR/` layout.

## Kaggle

Replace the Drive-mount cell with Kaggle's `/kaggle/input/<dataset>` for `DATA_DIR` and `/kaggle/working` for `OUTPUT_DIR`; everything else is unchanged.
