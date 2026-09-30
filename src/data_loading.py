"""Data loading: config, patch discovery, metadata and the pixel-level feature table.

Mirrors Task A of notebooks/exploration_and_training.ipynb.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import yaml

BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path="configs/config.yaml"):
    """Read the YAML config and resolve every entry under `paths` against the project root."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    cfg["class_names"] = {int(k): v for k, v in cfg["class_names"].items()}
    return cfg


def data_paths(cfg):
    """Return (S2 folder, annotation folder, metadata file) for the configured data dir."""
    base = cfg["paths"]["data_dir"]
    return base / "DATA_S2", base / "ANNOTATIONS", base / "metadata.geojson"


def _ids_in(folder, prefix):
    """Patch IDs found in file names like <prefix>_<id>.npy."""
    ids = set()
    for f in Path(folder).glob(f"{prefix}_*.npy"):
        m = re.match(rf"{prefix}_(\d+)\.npy", f.name)
        if m:
            ids.add(int(m.group(1)))
    return ids


def get_patch_ids(s2_dir, ann_dir):
    """Patch IDs that have BOTH a Sentinel-2 file and a label file, sorted."""
    s2_ids, ann_ids = _ids_in(s2_dir, "S2"), _ids_in(ann_dir, "TARGET")
    missing = (s2_ids ^ ann_ids)
    if missing:
        print(f"Warning: {len(missing)} patches have only an image or only a label; they are skipped.")
    return sorted(s2_ids & ann_ids)


def load_metadata(meta_fp, patch_ids):
    """Metadata table indexed by patch ID, with a parsed `dates` column (DatetimeIndex per patch)."""
    with open(meta_fp) as f:
        meta = json.load(f)
    df = pd.DataFrame([ft["properties"] for ft in meta["features"]])
    df = df.set_index("ID_PATCH").loc[patch_ids].copy()

    def parse_dates(d):
        vals = [d[k] for k in sorted(d, key=int)]
        return pd.to_datetime([str(v) for v in vals], format="%Y%m%d")

    df["dates"] = df["dates-S2"].apply(parse_dates)
    return df


def load_patch(s2_dir, ann_dir, pid, scale=10000.0):
    """One patch: reflectance array (T, C, H, W) float32 and label map (H, W) uint8."""
    s2 = np.load(Path(s2_dir) / f"S2_{pid}.npy").astype(np.float32) / scale
    lab = np.load(Path(ann_dir) / f"TARGET_{pid}.npy")[0]
    return s2, lab


def build_pixel_table(s2_dir, ann_dir, patch_ids, scale=10000.0):
    """Stack every pixel of every patch into one table.

    Returns
    -------
    X : (n_pixels, T*C) float32   one row per pixel, columns = all dates x all bands
    y : (n_pixels,) uint8         class ID per pixel
    patch_of_pixel : (n_pixels,)  patch ID of each row, used for patch-level splitting
    """
    T, C, H, W = np.load(Path(s2_dir) / f"S2_{patch_ids[0]}.npy", mmap_mode="r").shape
    n = len(patch_ids) * H * W
    X = np.empty((n, T * C), dtype=np.float32)
    y = np.empty(n, dtype=np.uint8)
    patch_of_pixel = np.empty(n, dtype=np.int32)

    for i, pid in enumerate(patch_ids):
        s2, lab = load_patch(s2_dir, ann_dir, pid, scale)
        a, b = i * H * W, (i + 1) * H * W
        # (T, C, H, W) -> (H, W, T, C) -> (H*W, T*C)
        X[a:b] = s2.transpose(2, 3, 0, 1).reshape(H * W, T * C)
        y[a:b] = lab.ravel()
        patch_of_pixel[a:b] = pid
    return X, y, patch_of_pixel


def load_dataset(cfg):
    """Convenience wrapper: patch IDs, metadata and the full pixel table in one call."""
    s2_dir, ann_dir, meta_fp = data_paths(cfg)
    patch_ids = get_patch_ids(s2_dir, ann_dir)
    df_meta = load_metadata(meta_fp, patch_ids)
    X, y, patch_of_pixel = build_pixel_table(s2_dir, ann_dir, patch_ids, cfg["data"]["reflectance_scale"])
    print(f"Loaded {len(patch_ids)} patches -> X {X.shape} ({X.nbytes / 1e9:.2f} GB)")
    return patch_ids, df_meta, X, y, patch_of_pixel
