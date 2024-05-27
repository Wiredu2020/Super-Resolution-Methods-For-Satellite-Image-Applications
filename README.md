# Super-Resolution Methods for Satellite Image Applications

AIMS MSc Thesis — reconstructing high-resolution satellite imagery from low-resolution inputs using deep learning.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Django-092E20?style=flat&logo=django&logoColor=white" alt="Django">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

## Overview

Low-cost satellite sensors often trade resolution for coverage, limiting how useful the imagery is for downstream analysis. This thesis implements and compares deep learning approaches for single-image super-resolution on satellite imagery, benchmarked against classical (non-learning) enhancement methods.

## Models

| Model | Description |
|---|---|
| **SRCNN** | Super-Resolution Convolutional Neural Network — baseline deep learning approach |
| **MSRCNN** | Modified SRCNN with an added upsampling layer and a Sigmoid activation, proposed to improve visual quality |
| **EDSR** | Enhanced Deep Super-Resolution network |
| **SRGAN** | Super-Resolution Generative Adversarial Network |

Classical interpolation-based methods are also implemented for comparison.

## Key Finding

Traditional image enhancement techniques underperform deep learning methods on satellite imagery. Among the models evaluated, **SRGAN** produced the best perceptual quality for resolution enhancement.

## Repository Structure

```
Models/                  Model implementations (notebooks, weights*, results)
Data Preprocessing/      Dataset preparation notebooks
Web_APP/                 Django demo app for running inference
Bright_Wiredu_Nuakoh_Thesis_2023-24/   Thesis manuscript (LaTeX)
```

\* Trained weights, checkpoints, and raw datasets are not tracked in this repository (see `.gitignore`) — only code and result images are versioned.

## Author

**Bright Wiredu Nuakoh** — African Institute for Mathematical Sciences (AIMS)
