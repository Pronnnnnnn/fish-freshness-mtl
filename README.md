# Fish Freshness MTL

Joint classification of fish species and freshness level from eye images, using
a multi-task convolutional network (hard parameter sharing, EfficientNetV2-B0
backbone) with a controlled comparison of three loss-weighting strategies:
Equal Weighting (EW), Uncertainty Weighting (UW, Kendall et al. 2018), and
Dynamic Weight Averaging (DWA, Liu et al. 2019).

## Data

[Freshness of the Fish Eyes (FFE)](https://doi.org/10.17632/xzyx7pbr3w.1)
(Prasetyo, Adityo, and Suciati, 2022), 4,390 images across 8 species and 3
freshness levels, released under CC BY 4.0. Not included in this repository;
see [Setup](#setup) for how to obtain and mount it.

```bibtex
@misc{prasetyo2022ffe,
  author       = {Prasetyo, Eko Cahyo Fanani and Adityo, Raden Dimas and Suciati, Nanik},
  title        = {The Freshness of the Fish Eyes Dataset},
  year         = {2022},
  publisher    = {Mendeley Data},
  doi          = {10.17632/xzyx7pbr3w.1}
}
```

## Repository structure

```
01_Dataset/       raw FFE images (not committed, see .gitignore)
02_Manifests/     label manifest and train/val/test split CSVs
03_Notebooks/     Colab notebooks, one per pipeline stage
04_Src/           reusable modules imported by the notebooks
05_Checkpoints/   trained weights (not committed, stored on Drive)
06_Results/       training logs, figures, metric summaries
07_Paper/         manuscript draft and figures
```

## Models compared

| Model | Heads | Loss |
|---|---|---|
| A | species only | cross-entropy |
| B | freshness only | cross-entropy |
| C-EW | species + freshness | equal weighting |
| C-UW | species + freshness | uncertainty weighting |
| C-DWA | species + freshness | dynamic weight averaging |

## Setup

1. Zip `01_Dataset/8_fish_3_freshness/` and upload it to
   `Google Drive/fish-freshness-mtl/8_fish_3_freshness.zip`.
2. Open the notebooks below in Colab, in order:

| # | Notebook | Purpose |
|---|---|---|
| 1 | `01_dataset_verification.ipynb` | integrity check, build label manifest |
| 2 | `02_train_val_test_split.ipynb` | stratified 70/15/15 split |
| 3 | `03_train_all_experiments.ipynb` | train all 5 models x 3 seeds |
| 4 | `04_test_set_evaluation.ipynb` | test-set metrics, aggregated over seeds |
| 5 | `05_gradcam_analysis.ipynb` | per-head Grad-CAM on the best model |

Each notebook clones this repository at the top of its first cell and is
runnable independently -- Colab does not persist state between notebooks, so
none of them assume an earlier notebook already ran in the same session.
Checkpoints and training history are written to Drive rather than the
notebook's local clone, so an interrupted session can resume by re-running
the same notebook.

## References

- Kendall, Gal, Cipolla. "Multi-Task Learning Using Uncertainty to Weigh
  Losses for Scene Geometry and Semantics." CVPR 2018.
- Liu, Johns, Davison. "End-to-End Multi-Task Learning with Attention."
  CVPR 2019.
- Tan, Le. "EfficientNetV2: Smaller Models and Faster Training." ICML 2021.
- Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization." ICCV 2017.
