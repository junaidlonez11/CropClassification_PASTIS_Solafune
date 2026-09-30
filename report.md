# Crop Type Classification — Report

## 1. Data and area of interest

The subset has 102 Sentinel-2 patches of 128 x 128 pixels, each with 46 acquisitions from 2018-09-20 to 2019-10-25 and 10 spectral bands. All patches share the same dates. The patches lie in a single Sentinel-2 tile (T31TFM), within a ~33 x 36 km box centered near 46.77° N, 5.16° E in eastern France (Saône-et-Loire / Jura area).

The label maps contain 19 of the 20 PASTIS classes (Beet is absent). The distribution is highly imbalanced: Background (30%), Meadow (18%), Soft winter wheat (12%) and Corn (12%) dominate, while Grapevine, Potatoes and Orchard occur in a single patch each. About 8.6% of pixels are Void. The one-year window covers both winter crops (sown in autumn, harvested in summer: wheat, barley, rapeseed) and summer crops (sown in spring, harvested in autumn: corn, soybeans, sunflower, sorghum), which is consistent with a temperate European rotation.

## 2. Method

**Features.** Each pixel is one sample with 460 features (46 dates x 10 bands), as reflectance (raw / 10000). All dates are kept; no cloud masking or compositing is applied.

**Split.** The split is at patch level, using the provided spatial `Fold` column: train folds 1–3 (66 patches), validation fold 4 (18) and test fold 5 (18). Splitting by pixel would leak information between neighbouring, correlated pixels.

**Model.** Random Forest, chosen as a strong, interpretable baseline that runs on a CPU (8 cores, no GPU) and needs no feature scaling: 200 trees, `max_depth=20`, `min_samples_leaf=5`, `class_weight="balanced"` for the class imbalance. Void pixels are removed. Training on ~988 K pixels takes ~36 minutes.

**Cross-validation.** The fixed split leaves some classes out of evaluation: Grapevine, Potatoes and Orchard are absent from both validation and test, and Sorghum is absent from validation. A rotating 5-fold cross-validation was therefore added. In each round one fold is the test set, the previous fold is validation and the other three are training, so every patch is tested once. Leave-one-patch-out was considered but rejected: 102 training runs (~20x the cost), very noisy single-patch test sets, and no natural validation set.

## 3. Results

**Fixed split (test = fold 5)**

| Metric | Validation | Test |
|---|---|---|
| Overall accuracy | 0.852 | 0.839 |
| Macro F1 | 0.51 | 0.49 |
| mIoU | 0.45 | 0.43 |
| MCC | 0.819 | 0.798 |

| Test, by class support | Classes | Macro F1 | Macro IoU |
|---|---|---|---|
| Well-represented (≥25 patches) | 9 | 0.716 | 0.642 |
| Sparse (<25 patches) | 6 | 0.143 | 0.125 |

**5-fold cross-validation (test fold rotated; round 5 = the fixed split)**

| Round | Train folds | Val fold | Test fold | Test accuracy | Test macro F1 | Test mIoU | Test MCC |
|---|---|---|---|---|---|---|---|
| 1 | 2, 3, 4 | 5 | 1 | 0.816 | 0.490 | 0.423 | 0.769 |
| 2 | 3, 4, 5 | 1 | 2 | 0.831 | 0.390 | 0.341 | 0.785 |
| 3 | 1, 4, 5 | 2 | 3 | 0.864 | 0.520 | 0.458 | 0.831 |
| 4 | 1, 2, 5 | 3 | 4 | 0.856 | 0.519 | 0.463 | 0.823 |
| 5 | 1, 2, 3 | 4 | 5 | 0.839 | 0.487 | 0.435 | 0.798 |
| **Mean ± std** | | | | **0.841 ± 0.019** | **0.481 ± 0.053** | **0.424 ± 0.049** | **0.801 ± 0.026** |

Validation means are almost identical (accuracy 0.840 ± 0.018, macro F1 0.481 ± 0.056, mIoU 0.424 ± 0.049, MCC 0.800 ± 0.024).

| Pooled out-of-fold, by class support | Classes | Macro precision | Macro recall | Macro F1 | Macro IoU |
|---|---|---|---|---|---|
| Well-represented (≥25 patches) | 9 | 0.856 | 0.721 | 0.729 | 0.642 |
| Sparse (<25 patches) | 9 | 0.211 | 0.086 | 0.101 | 0.081 |
| All classes | 18 | 0.533 | 0.403 | 0.415 | 0.362 |

The pooled predictions cover all 1,527,081 non-void pixels, each predicted once by a model that never saw its patch (pooled accuracy 0.841, MCC 0.803).

## 4. Interpretation

Accuracy is similar on validation and test, so the model generalises to unseen patches rather than overfitting to the training locations. Cross-validation confirms this: fold 5 (0.839 accuracy, 0.487 macro F1) sits almost exactly on the 5-fold mean (0.841 ± 0.019, 0.481 ± 0.053), so the fixed split was representative rather than lucky. Accuracy and MCC are high because they are dominated by the large classes. The macro metrics are much lower because every class counts equally, and the class-support table shows why: classes found in 25 or more patches reach a macro IoU of 0.64, while sparse classes reach only 0.13. The main limitation is data availability, not the model itself.

Across folds, accuracy and MCC barely move (std 0.02–0.03), but macro F1 ranges from 0.39 (test fold 2) to 0.52 (folds 3 and 4). The macro scores depend on which rare classes happen to fall in the test fold, since each rare class counts as much as a major one. The pooled out-of-fold results show the same split as the fixed test set: well-represented classes reach macro IoU 0.642, identical to fold 5, while sparse classes drop to 0.081. The sparse score is lower than on fold 5 (0.125) because the pooled set also includes Grapevine, Potatoes and Orchard, which can never be predicted. Sparse-class precision (0.21) is well above recall (0.09): the model seldom predicts these classes and misses most of their pixels.

**Classes that performed well.** Winter rapeseed (IoU ~0.91), Corn (~0.87), Soft winter wheat (~0.83–0.88), Winter barley (~0.86) and Soybeans (~0.83). All are common and have distinctive seasonal profiles: rapeseed's spring flowering, and corn and soybeans' late-summer peak.

**Classes that performed poorly.** Spring barley, Winter durum wheat, Fruits/vegetables/flowers, Mixed cereal and Sorghum (IoU near 0), and Leguminous fodder (~0.1). Grapevine, Potatoes and Orchard exist in only one patch each; in the fixed split all three are in training only, and in cross-validation the model has no training examples of them in the round where they are tested.

**Confusions.** Winter triticale is mostly predicted as Soft winter wheat: both are winter cereals with near-identical calendars and reflectance. Leguminous fodder is confused with Meadow, and Sorghum with Soybeans; these are green vegetative covers with less distinct temporal signatures than the major cereals and oilseeds.

**Class imbalance and limited samples.** `class_weight="balanced"` gives rare classes more weight during training, but cannot create variety that is not in the data. A class seen in 1–3 patches has almost no variation in field size, soil or timing to learn from. Label noise at field edges and in mixed pixels likely adds to the errors for small-field classes.

**Resolution, clouds and timing.** 10 m pixels mix neighbouring fields at boundaries, and the 20 m bands (red edge, SWIR) are resampled, so they are blurrier than the grid suggests. Clouds are not masked (see the cloud check in B.10 of the notebook), which adds noise to some dates; the full time series still provides enough clear observations for the major classes. All patches share the same 46 dates, so there are no missing observations between patches, and the year-long window covers the full crop calendar.

**Spatial context.** The model classifies each pixel independently, which produces speckle and ragged boundaries in the predicted maps (Task F).

## 5. Next steps

1. **More data for rare classes.** This would improve the macro scores more than any modelling change.
2. **Spatial context.** Add neighbourhood or texture features, or post-process predictions with a majority filter within fields. Alternatively, move to a segmentation model such as U-Net or a temporal model such as U-TAE on GPU.
3. **Cloud handling.** Use a cloud mask or per-date quality flag, and down-weight or interpolate cloudy observations.
4. **Compact temporal features.** Features such as NDVI and other indices, monthly composites and phenology metrics would shrink the 460-feature input and may help separate the similar winter cereals.
5. **Model selection with cross-validation.** Tune hyperparameters on each round's validation fold.
