"""Evaluate the saved model on the fixed split, and optionally run the rotating 5-fold CV (Task E).

Usage (from the project root):
    python -m src.evaluate                  # fixed split (val = fold 4, test = fold 5)
    python -m src.evaluate --cv             # also run the rotating 5-fold cross-validation (~5x training time)
"""
import argparse
import gc

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    jaccard_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from src.data_loading import load_config, load_dataset
from src.preprocessing import fixed_split, rotating_splits, save_split_ids, select_rows
from src.train import build_model
from src.visualization import plot_confusion_matrix


# ----------------------------------------------------------------------------- metrics
def binary_mcc(y_true, y_pred, c):
    """One-vs-rest MCC for class c, from counts (float64 avoids overflow on large pixel counts)."""
    t, p = (y_true == c), (y_pred == c)
    tp, fp, fn = (np.float64(np.sum(a)) for a in (t & p, ~t & p, t & ~p))
    tn = np.float64(t.size) - tp - fp - fn
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom


def summary_metrics(y_true, y_pred):
    """Overall accuracy, macro P/R/F1/IoU and multiclass MCC."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p, "macro_recall": r, "macro_f1": f1,
        "macro_IoU": jaccard_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def per_class_table(y_true, y_pred, classes, class_names, patches_present=None):
    """Per-class precision, recall, F1 and IoU."""
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    iou = jaccard_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    df = pd.DataFrame({"class_id": classes, "name": [class_names.get(c, "?") for c in classes],
                       "pixels": sup, "precision": p, "recall": r, "f1": f1, "IoU": iou})
    if patches_present is not None:
        df.insert(2, "patches_present", [patches_present.get(c, 0) for c in classes])
    return df.round(4)


def group_table(y_true, y_pred, classes, patches_present, min_patches):
    """Macro metrics for well-represented vs sparse classes (by number of patches containing them)."""
    groups = {
        f"Well-represented (>={min_patches} patches)": [c for c in classes if patches_present.get(c, 0) >= min_patches],
        f"Sparse (<{min_patches} patches)": [c for c in classes if patches_present.get(c, 0) < min_patches],
        "All classes": list(classes),
    }
    rows = []
    for name, g in groups.items():
        if not g:
            continue
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=g, average="macro", zero_division=0)
        rows.append({"group": name, "n_classes": len(g), "macro_precision": p, "macro_recall": r, "macro_f1": f1,
                     "macro_IoU": jaccard_score(y_true, y_pred, labels=g, average="macro", zero_division=0),
                     "macro_MCC": np.mean([binary_mcc(y_true, y_pred, c) for c in g])})
    return pd.DataFrame(rows).round(4)


def patches_per_class(y, patch_of_pixel):
    """Number of patches each class appears in."""
    df = pd.DataFrame({"c": y, "p": patch_of_pixel}).drop_duplicates()
    return df.groupby("c")["p"].nunique().to_dict()


# ----------------------------------------------------------------------------- runs
def evaluate_fixed_split(cfg, df_meta, X, y, patch_of_pixel, patches_present):
    """Task E on the fixed split, using the model saved by src.train."""
    void, names, out = cfg["data"]["void_class"], cfg["class_names"], cfg["paths"]["metrics_dir"]
    rf = joblib.load(cfg["paths"]["model_path"])
    split = fixed_split(df_meta, cfg)

    rows = []
    for name in ["val", "test"]:
        Xs, ys = select_rows(X, y, patch_of_pixel, split[name], void)
        yp = rf.predict(Xs)
        rows.append({"split": name, **summary_metrics(ys, yp)})
        classes = sorted(np.unique(ys).tolist())
        per_class_table(ys, yp, classes, names, patches_present).to_csv(out / f"fixed_split_per_class_{name}.csv", index=False)
        print(f"\n{name.upper()}\n" + classification_report(
            ys, yp, labels=classes, target_names=[f"{c}:{names.get(c, '?')}" for c in classes], zero_division=0))
        if name == "test":
            cm = confusion_matrix(ys, yp, labels=classes)
            labels = [names.get(c, "?") for c in classes]
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / "confusion_matrix_test.csv")
            plot_confusion_matrix(cm, labels, cfg["paths"]["figures_dir"] / "confusion_matrix_test.png")
            group_table(ys, yp, classes, patches_present, cfg["evaluation"]["well_represented_min_patches"]) \
                .to_csv(out / "fixed_split_metrics_by_class_support.csv", index=False)

    df = pd.DataFrame(rows).round(4)
    df.to_csv(out / "fixed_split_summary.csv", index=False)
    print(df.to_string(index=False))


def run_cross_validation(cfg, df_meta, X, y, patch_of_pixel, patches_present):
    """Rotating 5-fold CV: retrain in each round, score val and test, pool the test predictions."""
    void, out = cfg["data"]["void_class"], cfg["paths"]["metrics_dir"]
    rows, oof_true, oof_pred = [], [], []

    for rnd, split, roles in rotating_splits(df_meta, cfg["split"]["all_folds"]):
        save_split_ids(split, cfg["paths"]["splits_dir"] / "cv", prefix=f"round{rnd}_")
        X_tr, y_tr = select_rows(X, y, patch_of_pixel, split["train"], void)
        rf = build_model(cfg).fit(X_tr, y_tr)
        del X_tr, y_tr

        for name in ["val", "test"]:
            Xs, ys = select_rows(X, y, patch_of_pixel, split[name], void)
            yp = rf.predict(Xs)
            rows.append({"round": rnd, "split": name, **roles, **summary_metrics(ys, yp)})
            if name == "test":
                oof_true.append(ys)
                oof_pred.append(yp)
        print(f"Round {rnd}: {roles}")
        del rf
        gc.collect()

    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "cv_per_round_metrics.csv", index=False)
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_IoU", "MCC"]
    summary = df.groupby("split")[metrics].agg(["mean", "std"]).round(4)
    summary.to_csv(out / "cv_summary_mean_std.csv")
    print("\nMean ± std across rounds:\n" + summary.to_string())

    # Pooled out-of-fold predictions: every non-void pixel scored once
    yt, yp = np.concatenate(oof_true), np.concatenate(oof_pred)
    classes = sorted(np.unique(yt).tolist())
    per_class_table(yt, yp, classes, cfg["class_names"], patches_present).to_csv(out / "cv_oof_per_class_metrics.csv", index=False)
    grp = group_table(yt, yp, classes, patches_present, cfg["evaluation"]["well_represented_min_patches"])
    grp.to_csv(out / "cv_oof_metrics_by_class_support.csv", index=False)
    print("\nPooled out-of-fold, by class support:\n" + grp.to_string(index=False))


def main(config_path, cv):
    cfg = load_config(config_path)
    cfg["paths"]["metrics_dir"].mkdir(parents=True, exist_ok=True)
    cfg["paths"]["figures_dir"].mkdir(parents=True, exist_ok=True)
    patch_ids, df_meta, X, y, patch_of_pixel = load_dataset(cfg)
    patches_present = patches_per_class(y, patch_of_pixel)

    evaluate_fixed_split(cfg, df_meta, X, y, patch_of_pixel, patches_present)
    if cv:
        run_cross_validation(cfg, df_meta, X, y, patch_of_pixel, patches_present)
    print(f"\nMetrics saved to {cfg['paths']['metrics_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--cv", action="store_true", help="also run the rotating 5-fold cross-validation")
    args = parser.parse_args()
    main(args.config, args.cv)
