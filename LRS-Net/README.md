# LRS-Net — Colab/Kaggle Training

GPU training for LRS-Net (Lightweight Remote-Sensing Super-Resolution Network). CPU-only work — model comparison, the Pareto plot, downstream-task evaluation — happens locally afterward using the outputs this produces.

## Structure

```
LRS-Net/
  LRSNet_Training.ipynb   Run top to bottom on Colab or Kaggle (auto-detects which)
  lrsnet/                 Model, data pipeline, WorldStrat adapter, losses, metrics, plotting
  requirements.txt
```

`lrsnet/` has no Colab/Kaggle-specific calls (except `worldstrat.py`, which needs `rasterio`), so it also imports locally later for CPU-side work.

## Data

Expected layout — flat, no pre-split subfolders. The notebook splits train/val/test in code (`data.load_split_datasets`, seeded, saved to `split.json` for reuse):

```
DATA_DIR/
  HR/*.png
  LR/*.png   # omit entirely (LR_DIR = None) to use synthetic bicubic degradation instead
```

HR and LR filenames must match exactly (`list_pairs` pairs by filename, not just sort order — a mismatch raises instead of silently mispairing).

The RSA data (`Processed_Data/`) is already in this layout (2,202 matched HR/LR pairs).

### Real LR/HR pairs (vs. the synthetic bicubic downsampling used elsewhere in this thesis)

| Dataset | Pairing | Resolution | Notes |
|---|---|---|---|
| [WorldStrat](https://arxiv.org/abs/2207.06418) | Real: Airbus SPOT 6/7 (HR) + temporally-matched Sentinel-2 (LR) | 1.5 m / 10 m | Ready-to-use on Kaggle: [`jucor1/worldstrat`](https://www.kaggle.com/datasets/jucor1/worldstrat), add via "+ Add Input", no download script needed. HR imagery is CC-BY-NC 4.0 (fine for thesis use). Multi-band 12-bit GeoTIFFs — the notebook's §2b converts these to the flat RGB-PNG layout above via `lrsnet/worldstrat.py` (needs `rasterio`) |
| [SEN2VENµS](https://www.nature.com/articles/s41597-023-02538-9) | Real: Sentinel-2 + VENµS, same-day acquisitions | 5 m / 10-20 m | 132,955 patches, 29 locations, very clean pairing (≤30 min apart) |
| [OLI2MSI](https://github.com/wjwjww/OLI2MSI) | Real: Landsat-8 OLI (LR) + Sentinel-2 MSI (HR) | 30 m / 10 m | Smallest/simplest to get started with, fixed x3 scale |

WorldStrat is the one with notebook support (§2b) since it's directly available as a ready-to-use Kaggle dataset. SEN2VENµS/OLI2MSI would need a similar conversion step written for their own formats before pointing `HR_DIR`/`LR_DIR` at them.

`lrsnet/worldstrat.py`'s file-discovery patterns (`hr_glob`/`lr_glob`) are a best guess from the paper, not verified against a live mount — run `worldstrat.inspect(root)` first and adjust them if they don't match what's actually there.

## Outputs

Written to `OUTPUT_DIR`:

```
weights/lrsnet_best.weights.h5
weights/lrsnet_final.weights.h5
figures/lrsnet_loss_psnr.png
figures/lrsnet_val_predictions.png
history.csv
model_report.json
split.json          # exact train/val/test file lists -- reuse "test" for later comparisons
```

Copy this folder into `Models/LRSNet/` locally when done, matching the existing `Models/SRCNN/`, `Models/EDSR/` layout.

## Kaggle vs. Colab

The notebook detects which platform it's on (`/kaggle/working` exists or not) and sets `DATA_DIR`/`OUTPUT_DIR` accordingly — no manual editing needed either way. On Kaggle: add your dataset via "+ Add Input" and turn on internet access in Settings (needed for the `git clone` cell).
