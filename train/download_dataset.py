"""
download_dataset.py — Dataset Downloader & Extractor

Downloads the Face Mask Detection dataset from:
  https://github.com/chandrikadeb7/Face-Mask-Detection

and organises images into:
  data/with_mask/       → images of faces WITH masks
  data/without_mask/    → images of faces WITHOUT masks

Usage
-----
  python train/download_dataset.py
  python train/download_dataset.py --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import get_config, TrainingConfig


# ─────────────────────────────────────────────────────────────────────────────
# Download helper
# ─────────────────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path, chunk_size: int = 65536) -> None:
    """Download `url` to `dest` with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    desc = dest.name[:40]

    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset organiser
# ─────────────────────────────────────────────────────────────────────────────

def _count_images(directory: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for f in directory.rglob("*") if f.suffix.lower() in exts)


def _copy_images(src: Path, dst: Path) -> int:
    """Copy all images from src (recursively) to dst. Returns count."""
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for img in src.rglob("*"):
        if img.suffix.lower() in exts:
            shutil.copy2(img, dst / img.name)
            count += 1
    return count


def organise_dataset(zip_path: Path, data_dir: Path) -> None:
    """Extract zip and arrange images into data/with_mask / without_mask."""
    extract_dir = zip_path.parent / "raw_extracted"
    print(f"\n[INFO] Extracting {zip_path.name} ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # The GitHub ZIP contains: Face-Mask-Detection-master/dataset/with_mask/ etc.
    # Try to find the subdirectory layout
    dataset_root = None
    for candidate in extract_dir.rglob("with_mask"):
        if candidate.is_dir():
            dataset_root = candidate.parent
            break

    if dataset_root is None:
        # Fallback: look for any image-containing subdirectories
        print("[WARN] Standard layout not found. Scanning for image folders...")
        all_dirs = [d for d in extract_dir.rglob("*") if d.is_dir() and _count_images(d) > 0]
        # Heuristic: folders with "mask" vs "no_mask" / "without"
        for d in all_dirs:
            name_lower = d.name.lower()
            if "without" in name_lower or "no_mask" in name_lower or "nomask" in name_lower:
                print(f"  Copying {d} → data/without_mask")
                _copy_images(d, data_dir / "without_mask")
            elif "with" in name_lower or "mask" in name_lower:
                print(f"  Copying {d} → data/with_mask")
                _copy_images(d, data_dir / "with_mask")
    else:
        print(f"  Found dataset root: {dataset_root}")
        with_src = dataset_root / "with_mask"
        without_src = dataset_root / "without_mask"

        n1 = _copy_images(with_src,    data_dir / "with_mask")
        n2 = _copy_images(without_src, data_dir / "without_mask")
        print(f"  Copied {n1} with_mask images")
        print(f"  Copied {n2} without_mask images")

    # Cleanup extract dir
    shutil.rmtree(extract_dir, ignore_errors=True)
    print("[OK] Dataset ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dataset(cfg: TrainingConfig) -> None:
    """
    Ensure the dataset exists in cfg.data_dir.
    Downloads and extracts if not already present.
    """
    data_dir = Path(cfg.data_dir)
    with_mask_dir = data_dir / "with_mask"
    without_mask_dir = data_dir / "without_mask"

    n_with = _count_images(with_mask_dir) if with_mask_dir.exists() else 0
    n_without = _count_images(without_mask_dir) if without_mask_dir.exists() else 0

    if n_with >= 100 and n_without >= 100:
        print(f"[OK] Dataset already present ({n_with} with_mask / {n_without} without_mask). Skipping download.")
        return

    zip_path = Path(cfg.dataset_zip)
    if not zip_path.exists():
        print(f"\n[INFO] Downloading dataset from:\n   {cfg.dataset_url}")
        _download(cfg.dataset_url, zip_path)
        print(f"   Saved to: {zip_path}")
    else:
        print(f"[OK] Dataset zip already downloaded: {zip_path}")

    organise_dataset(zip_path, data_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download & prepare mask dataset")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = get_config(args.config)
    ensure_dataset(cfg.training)


if __name__ == "__main__":
    main()
