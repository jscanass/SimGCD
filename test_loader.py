#!/usr/bin/env python3
"""
Quick sanity check: build batdetect2 datasets via get_datasets and print first-batch shapes.

Run from the SimGCD directory (so `data/` imports resolve), e.g.:
  cd ch3/models/SimGCD && python test_loader.py --batdetect2_csv_path /path/to/annotations.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import torch
from torch.utils.data import DataLoader

# SimGCD root (directory containing this file)
_SIMGCD_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SIMGCD_ROOT not in sys.path:
    sys.path.insert(0, _SIMGCD_ROOT)

from config import osr_split_dir  # noqa: E402
from data.augmentations import get_spectrogram_transform  # noqa: E402
from data.get_datasets import get_class_splits, get_datasets  # noqa: E402


def _default_csv() -> str:
    # SimGCD lives at <repo>/ch3/models/SimGCD
    repo_root = os.path.normpath(os.path.join(_SIMGCD_ROOT, "..", "..", ".."))
    return os.path.join(
        repo_root,
        "ch2",
        "datasets",
        "echolocation",
        "annotations",
        "batdetect2_echospfull.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test batdetect2 DataLoader shapes")
    parser.add_argument(
        "--batdetect2_csv_path",
        type=str,
        default=_default_csv(),
        help="Annotations CSV (must contain train/test rows and species_id)",
    )
    parser.add_argument(
        "--batdetect2_audio_root",
        type=str,
        default=None,
        help="Optional root prepended to relative audio_path in CSV",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    args_ns = parser.parse_args()

    # Minimal args namespace expected by get_class_splits / get_datasets
    class Args:
        pass

    args = Args()
    args.dataset_name = "batdetect2"
    args.prop_train_labels = 0.5
    args.batdetect2_csv_path = args_ns.batdetect2_csv_path
    args.batdetect2_audio_root = args_ns.batdetect2_audio_root

    # 18-class bio split: known 0–9, unknown 10–17 (same as bio18_class_splits.pkl)
    split_path = os.path.join(_SIMGCD_ROOT, osr_split_dir, "bio18_class_splits.pkl")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Missing split file: {split_path}")
    with open(split_path, "rb") as f:
        splits = pickle.load(f)
    args.train_classes = splits["Old"]
    args.unlabeled_classes = splits["New"]

    args = get_class_splits(args)

    train_transform, test_transform = get_spectrogram_transform(image_size=args.image_size)

    _, test_dataset, _, _ = get_datasets(
        "batdetect2",
        train_transform,
        test_transform,
        args,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=args_ns.batch_size,
        shuffle=False,
        num_workers=args_ns.num_workers,
        pin_memory=False,
    )

    images, targets, _ = next(iter(loader))

    print("images shape:", tuple(images.shape))
    print("targets shape:", tuple(targets.shape))
    print("targets dtype:", targets.dtype)

    expected = (args_ns.batch_size, 3, args.image_size, args.image_size)
    if tuple(images.shape) != expected:
        # Last batch could be smaller if dataset is tiny
        if images.shape[0] > args_ns.batch_size or images.shape[0] < 1:
            raise AssertionError(f"Unexpected batch dim: got {tuple(images.shape)}, expected {expected} (or smaller batch)")
        if tuple(images.shape[1:]) != (3, args.image_size, args.image_size):
            raise AssertionError(
                f"Expected image shape [*, 3, {args.image_size}, {args.image_size}], got {tuple(images.shape)}"
            )
    else:
        assert tuple(images.shape) == expected, f"Expected {expected}, got {tuple(images.shape)}"

    assert targets.shape[0] == images.shape[0], "targets batch must match images batch"
    print("OK: images are [Batch, 3, 224, 224] (with Batch <= --batch_size if dataset is short).")


if __name__ == "__main__":
    main()
