"""Figures and saved predictions (Tasks B and F).

Usage (from the project root), after src.train:
    python -m src.visualization            # test-patch figure + predicted maps for all test patches
"""
import argparse

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # scripts write files; no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.data_loading import data_paths, get_patch_ids, load_config, load_metadata, load_patch
from src.preprocessing import fixed_split


def rgb_clearest_date(s2, stretch=0.3):
    """True-colour image (B4, B3, B2) on the date with the lowest mean blue reflectance."""
    t = int(np.argmin(s2[:, 0].mean(axis=(1, 2))))
    return np.clip(s2[t, [2, 1, 0]].transpose(1, 2, 0) / stretch, 0, 1), t


def plot_label_patches(ann_dir, patch_ids, out_fp):
    """Label maps for a few patches (Task B.6)."""
    fig, axes = plt.subplots(1, len(patch_ids), figsize=(4 * len(patch_ids), 4))
    for ax, pid in zip(np.atleast_1d(axes), patch_ids):
        lab = np.load(ann_dir / f"TARGET_{pid}.npy")[0]
        im = ax.imshow(lab, cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
        ax.set_title(f"Patch {pid}", fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Class ID")
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, labels, out_fp):
    """Confusion matrix with a log colour scale, so rare classes remain visible."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", norm=LogNorm(vmin=1, vmax=max(cm.max(), 1)))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test set confusion matrix (log scale)")
    fig.colorbar(im, ax=ax, label="Pixel count (log)")
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150)
    plt.close(fig)


def predict_patch(rf, s2):
    """Per-pixel prediction for one patch, reshaped back to (H, W)."""
    T, C, H, W = s2.shape
    return rf.predict(s2.transpose(2, 3, 0, 1).reshape(H * W, T * C)).reshape(H, W).astype(np.uint8)


def main(config_path, n_show=4):
    cfg = load_config(config_path)
    s2_dir, ann_dir, meta_fp = data_paths(cfg)
    fig_dir, pred_dir = cfg["paths"]["figures_dir"], cfg["paths"]["predictions_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    patch_ids = get_patch_ids(s2_dir, ann_dir)
    df_meta = load_metadata(meta_fp, patch_ids)
    test_ids = fixed_split(df_meta, cfg)["test"]
    rf = joblib.load(cfg["paths"]["model_path"])
    scale = cfg["data"]["reflectance_scale"]

    plot_label_patches(ann_dir, patch_ids[:n_show], fig_dir / "label_patches.png")

    # --- Predict every test patch; save the maps; plot the first n_show (Task F) ---
    show = test_ids[:n_show]
    fig, axes = plt.subplots(len(show), 3, figsize=(12, 4 * len(show)), squeeze=False)
    for pid in test_ids:
        s2, lab = load_patch(s2_dir, ann_dir, pid, scale)
        pred = predict_patch(rf, s2)
        np.save(pred_dir / f"PRED_{pid}.npy", pred)
        if pid in show:
            row = show.index(pid)
            rgb, _ = rgb_clearest_date(s2)
            for ax, img, title in zip(axes[row], [rgb, lab, pred], [f"Patch {pid} — RGB", "Ground truth", "Prediction"]):
                if img.ndim == 2:
                    ax.imshow(img, cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
                else:
                    ax.imshow(img)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "test_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {len(test_ids)} predicted maps to {pred_dir} and figures to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    main(parser.parse_args().config)
