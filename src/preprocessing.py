"""Preprocessing and splitting: void removal, fold-based splits, rotating CV splits.

Mirrors Tasks C and D of notebooks/exploration_and_training.ipynb.
"""
from pathlib import Path

import numpy as np


def drop_void(X, y, void_class=19):
    """Remove void pixels (no real land-cover class) from a feature/label pair."""
    keep = y != void_class
    return X[keep], y[keep]


def ids_for_folds(df_meta, folds):
    """Sorted patch IDs whose `Fold` value is in `folds`."""
    return sorted(df_meta.index[df_meta["Fold"].isin(folds)].tolist())


def fixed_split(df_meta, cfg):
    """Task C split: dict of split name -> patch IDs, using the folds in the config."""
    s = cfg["split"]
    split = {
        "train": ids_for_folds(df_meta, s["train_folds"]),
        "val": ids_for_folds(df_meta, s["val_folds"]),
        "test": ids_for_folds(df_meta, s["test_folds"]),
    }
    _check_disjoint(split)
    return split


def rotating_splits(df_meta, folds=(1, 2, 3, 4, 5)):
    """Rotating 5-fold splits: in round k, fold k is test, the fold before it is val, the rest train.

    Every patch is in test once, in val once and in train three times across the rounds.
    Yields (round_number, split_dict, fold_roles).
    """
    folds = list(folds)
    n = len(folds)
    for k in range(n):
        test_fold = folds[k]                 # round k tests fold k; the last round equals the fixed split
        val_fold = folds[(k + n - 1) % n]    # the fold before the test fold
        train_folds = [f for f in folds if f not in (test_fold, val_fold)]
        split = {
            "train": ids_for_folds(df_meta, train_folds),
            "val": ids_for_folds(df_meta, [val_fold]),
            "test": ids_for_folds(df_meta, [test_fold]),
        }
        _check_disjoint(split)
        yield k + 1, split, {"train_folds": train_folds, "val_fold": val_fold, "test_fold": test_fold}


def _check_disjoint(split):
    """No patch may appear in more than one split."""
    names = list(split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(split[a]) & set(split[b])
            assert not overlap, f"Patches in both {a} and {b}: {sorted(overlap)[:5]}"


def select_rows(X, y, patch_of_pixel, patch_ids, void_class=19):
    """Pixel rows belonging to `patch_ids`, with void pixels removed."""
    mask = np.isin(patch_of_pixel, patch_ids)
    return drop_void(X[mask], y[mask], void_class)


def save_split_ids(split, out_dir, prefix=""):
    """Write one text file per split, one patch ID per line."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in split.items():
        with open(out_dir / f"{prefix}{name}_patch_ids.txt", "w") as f:
            f.write("\n".join(str(p) for p in ids))
