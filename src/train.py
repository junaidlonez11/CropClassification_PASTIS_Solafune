"""Train the Random Forest on the fixed split (Task D) and save it.

Usage (from the project root):
    python -m src.train --config configs/config.yaml
"""
import argparse
import time

import joblib
from sklearn.ensemble import RandomForestClassifier

from src.data_loading import load_config, load_dataset
from src.preprocessing import fixed_split, save_split_ids, select_rows


def build_model(cfg, verbose=0):
    """Random Forest with the hyperparameters from the config."""
    return RandomForestClassifier(**cfg["model"], verbose=verbose)


def main(config_path):
    cfg = load_config(config_path)
    void = cfg["data"]["void_class"]

    # --- Data and fixed patch-level split (Task A + C) ---
    patch_ids, df_meta, X, y, patch_of_pixel = load_dataset(cfg)
    split = fixed_split(df_meta, cfg)
    save_split_ids(split, cfg["paths"]["splits_dir"])
    print({k: len(v) for k, v in split.items()}, "patches per split; IDs saved to", cfg["paths"]["splits_dir"])

    # --- Training rows (void removed) ---
    X_tr, y_tr = select_rows(X, y, patch_of_pixel, split["train"], void)
    del X  # free the full table; training only needs the train rows

    # --- Fit ---
    rf = build_model(cfg, verbose=1)
    start = time.time()
    rf.fit(X_tr, y_tr)
    print(f"Training time: {(time.time() - start) / 60:.1f} min on {X_tr.shape[0]} pixels")

    # --- Save ---
    model_path = cfg["paths"]["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    main(parser.parse_args().config)
