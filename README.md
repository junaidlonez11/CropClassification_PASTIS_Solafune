# Crop Type Classification — PASTIS Subset

Pixel-level crop-type classification from multi-temporal Sentinel-2 imagery, using a Random Forest on the full 46-date x 10-band time series of each pixel. Built for the Solafune Geo Data Scientist assignment.

## Project structure

```
project-root/
├── README.md
├── report.md                         # short analysis report (Task G)
├── requirements.txt
├── configs/
│   └── config.yaml                   # paths, split folds, model hyperparameters, class names
├── notebooks/
│   └── exploration_and_training.ipynb   # full workflow, Tasks A–G, with outputs
├── src/
│   ├── data_loading.py               # config, patch discovery, metadata, pixel feature table
│   ├── preprocessing.py              # void removal, fold-based and rotating CV splits
│   ├── train.py                      # train and save the model (fixed split)
│   ├── evaluate.py                   # metrics on the fixed split, optional 5-fold CV
│   └── visualization.py              # figures and predicted label maps
├── splits/                           # patch IDs per split (train/val/test, and cv/ per round)
├── outputs/
│   ├── figures/
│   ├── metrics/                      # CSV metric tables
│   ├── predictions/                  # predicted 128x128 label maps (PRED_<patch_id>.npy)
│   └── models/                       # trained model — git-ignored, see below
└── data/                             # not in the repo — see "Data"
```

## Data

The dataset is not included in the repository. Place it like this:

```
data/PASTIS_subset/
├── DATA_S2/          S2_<patch_id>.npy       (46, 10, 128, 128) int16
├── ANNOTATIONS/      TARGET_<patch_id>.npy   (1, 128, 128) uint8
└── metadata.geojson
```

To keep the data somewhere else, either edit `paths.data_dir` in `configs/config.yaml` (scripts) or set the `PASTIS_DATA_DIR` environment variable (notebook).

## Environment setup

Developed with Python 3.8 on a local machine (8 CPU cores, no GPU).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Memory:** the full pixel table is about 3.1 GB (1.67 M pixels x 460 float32 features), and the train/val/test copies add to that. 

## How to run

**Option 1 — notebook (full workflow with explanations).** Start Jupyter from the project root and open `notebooks/exploration_and_training.ipynb`. Run all cells in order. It writes figures, metrics, predictions, split files and the model to the folders above.

**Option 2 — scripts.** From the project root:

```bash
python -m src.train                 # load data, fixed split, train, save model   (~36 min)
python -m src.evaluate              # fixed-split metrics + confusion matrix
python -m src.evaluate --cv         # ...plus rotating 5-fold cross-validation     (~3 h)
python -m src.visualization         # test-patch figure + predicted maps
```

All settings (paths, folds, hyperparameters) are read from `configs/config.yaml`.

## Reproducing the main results

The pipeline is deterministic: the split comes from the `Fold` column in `metadata.geojson`, and the Random Forest uses `random_state=0`. Running `src.train` then `src.evaluate` reproduces the fixed-split results in the notebook and `report.md`. The exact patch IDs for every split are in `splits/`.

**Trained model.** The saved model (`outputs/models/rf_model.joblib`) is ~765 MB, so it is not committed. Regenerate it with `python -m src.train` (or the Task D cells of the notebook).

## Approach

1. **Features:** each pixel is one sample, with all 46 dates x 10 bands as features (460 columns). Reflectance is obtained by dividing by 10000. No cloud filtering or compositing is applied.
2. **Split:** patch-level, using the provided spatial `Fold` column — train folds 1–3, validation fold 4, test fold 5. Splitting by patch, not pixel, avoids spatial leakage between neighbouring pixels.
3. **Model:** Random Forest (200 trees, `max_depth=20`, `min_samples_leaf=5`, `class_weight="balanced"`). Void pixels (class 19) are excluded from training and evaluation.
4. **Evaluation:** overall accuracy, per-class precision/recall/F1/IoU, mIoU, MCC and a confusion matrix, plus macro metrics for well-represented vs sparse classes.
5. **Cross-validation:** because the fixed split leaves some classes missing from validation and test, a rotating 5-fold cross-validation is also run. Each fold is the test set once, and results are reported as mean ± std and on the pooled out-of-fold predictions.

## Key assumptions

- Sentinel-2 values are L2A surface reflectance scaled by 10000 (confirmed from value ranges).
- All 46 dates are kept, including cloudy ones.
- Class 19 (Void) is unlabelled and excluded; class 0 (Background) is treated as a real class.
- The `Fold` column is a valid spatial partition, so patches in different folds are independent enough for evaluation.

## Limitations and known issues

- Per-pixel classification ignores spatial context, giving noisy field boundaries.
- Several classes occur in only 1–13 patches; Grapevine, Potatoes and Orchard occur in a single patch each and cannot be learned or evaluated reliably.
- Clouds are not masked, adding noise to some dates.
- The 20 m bands are resampled to the 10 m grid, so they carry less spatial detail than the grid suggests.

See `report.md` for the full analysis.
