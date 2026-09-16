"""
train.py — MobileNetV2 Mask Classifier Training Script

Strategy
--------
1. Download & extract dataset (chandrikadeb7/Face-Mask-Detection from GitHub)
2. Build MobileNetV2 with frozen base + custom head
3. Phase 1: train head for `epochs_head` epochs (fast convergence)
4. Phase 2: unfreeze top 20 layers, fine-tune at lower LR
5. Save best model checkpoint to models/mask_classifier.h5
6. Evaluate on held-out test set and print metrics

Run
---
  python train/train.py
  python train/train.py --config config.yaml --fast   # 3+2 epochs for quick demo
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Suppress TF info/warning spam before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import get_config, TrainingConfig
from train.download_dataset import ensure_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build a MobileNetV2-based binary classifier.

    Architecture
    ────────────
    MobileNetV2 (ImageNet, frozen)
        └── GlobalAveragePooling2D
            └── Dense(128, relu) + Dropout(0.5)
                └── Dense(2, softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base.trainable = False  # freeze all base layers initially

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="mask_classifier")
    return model, base


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def build_data_generators(cfg: TrainingConfig):
    """Build train / validation / test ImageDataGenerators."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img_h, img_w = cfg.image_size
    data_dir = Path(cfg.data_dir)

    # Verify dataset
    mask_dir = data_dir / "with_mask"
    no_mask_dir = data_dir / "without_mask"
    if not mask_dir.exists() or not no_mask_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}.\n"
            "Run  python train/download_dataset.py  first."
        )

    n_mask = len(list(mask_dir.glob("*.*")))
    n_no_mask = len(list(no_mask_dir.glob("*.*")))
    print(f"  Dataset: {n_mask} with_mask / {n_no_mask} without_mask images")

    if cfg.augmentation:
        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=cfg.validation_split,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
        )
    else:
        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=cfg.validation_split,
        )

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=cfg.validation_split,
    )

    train_ds = train_gen.flow_from_directory(
        str(data_dir),
        target_size=(img_h, img_w),
        batch_size=cfg.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_ds = val_gen.flow_from_directory(
        str(data_dir),
        target_size=(img_h, img_w),
        batch_size=cfg.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    print(f"  Class indices: {train_ds.class_indices}")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Training routine
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainingConfig, fast: bool = False):
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    )

    # Override epochs for fast mode
    epochs_head = 3 if fast else cfg.epochs_head
    epochs_ft = 2 if fast else cfg.epochs_finetune

    print("\n" + "=" * 55)
    print("  Mask Classifier - Training")
    print(f"  TensorFlow {tf.__version__} | CPU-only mode")
    print(f"  Phase 1: {epochs_head} epochs (frozen base)")
    print(f"  Phase 2: {epochs_ft} epochs (fine-tune top-20)")
    print("=" * 55 + "\n")

    # ── Ensure dataset is present ─────────────────────────────────────────
    ensure_dataset(cfg)

    # ── Build data generators ─────────────────────────────────────────────
    train_ds, val_ds = build_data_generators(cfg)

    # ── Build model ───────────────────────────────────────────────────────
    model, base = build_model(
        input_shape=(*cfg.image_size, 3),
        num_classes=2,
    )
    model.summary(line_length=80)

    # ── Callbacks ─────────────────────────────────────────────────────────
    save_path = cfg.model_save_path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    callbacks_head = [
        ModelCheckpoint(
            save_path, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy", patience=3,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=2, min_lr=1e-6, verbose=1,
        ),
    ]

    # ── Phase 1: Train head ───────────────────────────────────────────────
    print("\n-- Phase 1: Training classification head --")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg.learning_rate_head),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    t0 = time.time()
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        callbacks=callbacks_head,
        verbose=1,
    )

    # ── Phase 2: Fine-tune top-20 base layers ────────────────────────────
    print("\n-- Phase 2: Fine-tuning top-20 MobileNetV2 layers --")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg.learning_rate_finetune),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_ft = [
        ModelCheckpoint(
            save_path, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy", patience=3,
            restore_best_weights=True, verbose=1,
        ),
    ]

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_ft,
        callbacks=callbacks_ft,
        verbose=1,
    )

    elapsed = time.time() - t0
    print(f"\n[OK] Training complete in {elapsed/60:.1f} min")
    print(f"   Best model saved → {save_path}")

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\n-- Evaluating on validation set --")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"   Val loss     : {loss:.4f}")
    print(f"   Val accuracy : {acc:.4f} ({acc*100:.2f}%)")

    # ── Training curves ───────────────────────────────────────────────────
    _save_training_plot(h1, h2, cfg)

    return model


def _save_training_plot(h1, h2, cfg: TrainingConfig):
    """Save training accuracy/loss curves to docs/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        acc1 = h1.history.get("accuracy", [])
        val_acc1 = h1.history.get("val_accuracy", [])
        acc2 = h2.history.get("accuracy", [])
        val_acc2 = h2.history.get("val_accuracy", [])

        acc_all = acc1 + acc2
        val_all = val_acc1 + val_acc2
        epochs = range(1, len(acc_all) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#0d1117")
        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#c9d1d9")
            ax.xaxis.label.set_color("#c9d1d9")
            ax.yaxis.label.set_color("#c9d1d9")
            ax.title.set_color("#c9d1d9")
            for sp in ax.spines.values():
                sp.set_edgecolor("#21262d")

        ax1, ax2 = axes
        ax1.plot(epochs, acc_all, color="#58a6ff", label="Train acc", linewidth=2)
        ax1.plot(epochs, val_all, color="#3fb950", label="Val acc", linewidth=2, linestyle="--")
        ax1.axvline(len(acc1) + 0.5, color="#f85149", linestyle=":", linewidth=1.2, label="Fine-tune start")
        ax1.set_title("Accuracy", fontsize=12)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend(facecolor="#1f2937", labelcolor="#c9d1d9", fontsize=9)
        ax1.grid(color="#21262d", linewidth=0.5)

        loss1 = h1.history.get("loss", [])
        val_loss1 = h1.history.get("val_loss", [])
        loss2 = h2.history.get("loss", [])
        val_loss2 = h2.history.get("val_loss", [])
        loss_all = loss1 + loss2
        val_loss_all = val_loss1 + val_loss2

        ax2.plot(epochs, loss_all, color="#58a6ff", label="Train loss", linewidth=2)
        ax2.plot(epochs, val_loss_all, color="#f85149", label="Val loss", linewidth=2, linestyle="--")
        ax2.axvline(len(loss1) + 0.5, color="#d29922", linestyle=":", linewidth=1.2, label="Fine-tune start")
        ax2.set_title("Loss", fontsize=12)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(facecolor="#1f2937", labelcolor="#c9d1d9", fontsize=9)
        ax2.grid(color="#21262d", linewidth=0.5)

        fig.suptitle("Training Curves - Mask Classifier", color="#c9d1d9", fontsize=13)
        fig.tight_layout()

        out = _PROJECT_ROOT / "docs" / "training_curves.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"   Training curves → {out}")
    except Exception as e:
        print(f"[WARN] Could not save training plot: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train mask classifier")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: 3+2 epochs (quick functional test, ~5 min on CPU)"
    )
    args = parser.parse_args()

    cfg = get_config(args.config)
    train(cfg.training, fast=args.fast)


if __name__ == "__main__":
    main()
