import os
import csv
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from copy import deepcopy

from data.data_utils import subsample_instances

import batdetect2.api as bd2_api
from batdetect2.detector.parameters import DEFAULT_SPECTROGRAM_PARAMETERS, TARGET_SAMPLERATE_HZ
import batdetect2.utils.audio_utils as bd2_au


@dataclass(frozen=True)
class BatDetect2Row:
    file_name: str
    audio_path: str
    species_id: int
    start_time: float
    end_time: float
    low_freq: float
    high_freq: float


class BatDetect2(Dataset):
    """
    Dataset wrapper for BatDetect2-style echolocation call annotations.

    Expects a CSV with columns:
      - file_name
      - audio_path
      - species_id
      - start_time, end_time (seconds)
      - low_freq, high_freq (Hz)
      - split ('train' or 'test')

    Returns: (image, target, uq_idx)
      - image: PIL image (or transformed tensor if transform provided)
      - target: int (or target_transform(target))
      - uq_idx: unique index (stable within this dataset instance)
    """

    def __init__(
        self,
        csv_path: str,
        audio_root: Optional[str] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        *,
        spec_params: Optional[dict] = None,
        samplerate_hz: int = TARGET_SAMPLERATE_HZ,
        min_bbox_width_px: int = 2,
        min_bbox_height_px: int = 2,
    ):
        self.csv_path = csv_path
        self.audio_root = audio_root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.samplerate_hz = int(samplerate_hz)
        self.spec_params = dict(DEFAULT_SPECTROGRAM_PARAMETERS if spec_params is None else spec_params)

        self._resize_factor = float(self.spec_params["resize_factor"])
        self._fft_win_length_s = float(self.spec_params["fft_win_length"])
        self._fft_overlap = float(self.spec_params["fft_overlap"])

        self._max_freq_hz = float(self.spec_params["max_freq"])
        self._min_freq_hz = float(self.spec_params["min_freq"])
        self._max_freq_bin = int(round(self._max_freq_hz * self._fft_win_length_s))
        self._min_freq_bin = int(round(self._min_freq_hz * self._fft_win_length_s))

        self._spec_height_out = int(self.spec_params["spec_height"] * self._resize_factor)
        self._min_bbox_width_px = int(min_bbox_width_px)
        self._min_bbox_height_px = int(min_bbox_height_px)

        required = {
            "file_name",
            "audio_path",
            "species_id",
            "start_time",
            "end_time",
            "low_freq",
            "high_freq",
            "split",
        }
        split_name = "train" if train else "test"
        rows: list[BatDetect2Row] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV appears to have no header: {csv_path}")
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

            for r in reader:
                if str(r["split"]).lower() != split_name:
                    continue
                rows.append(
                    BatDetect2Row(
                        file_name=str(r["file_name"]),
                        audio_path=str(r["audio_path"]),
                        species_id=int(r["species_id"]),
                        start_time=float(r["start_time"]),
                        end_time=float(r["end_time"]),
                        low_freq=float(r["low_freq"]),
                        high_freq=float(r["high_freq"]),
                    )
                )

        if len(rows) == 0:
            raise ValueError(f"No rows found for split='{split_name}' in {csv_path}")

        self.data = rows
        self.uq_idxs = np.array(range(len(self.data)))

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_audio_path(self, audio_path: str) -> str:
        if os.path.isabs(audio_path):
            return audio_path
        if self.audio_root is None:
            return audio_path
        return os.path.join(self.audio_root, audio_path)

    def _row(self, idx: int) -> BatDetect2Row:
        return self.data[idx]

    def _spec_to_pil(self, spec_2d: np.ndarray) -> Image.Image:
        spec_2d = np.asarray(spec_2d, dtype=np.float32)
        vmin = float(spec_2d.min())
        vmax = float(spec_2d.max())
        if vmax <= vmin:
            img_u8 = np.zeros_like(spec_2d, dtype=np.uint8)
        else:
            img_u8 = (255.0 * (spec_2d - vmin) / (vmax - vmin)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_u8, mode="L").convert("RGB")

    def _time_to_x(self, t_s: float) -> int:
        x = bd2_au.time_to_x_coords(
            float(t_s),
            samplerate=float(self.samplerate_hz),
            window_duration=float(self._fft_win_length_s),
            window_overlap=float(self._fft_overlap),
        )
        return int(round(x * self._resize_factor))

    def _freq_to_y(self, f_hz: float) -> int:
        # BatDetect2 spectrograms are cropped to [min_freq, max_freq] and (after cropping)
        # y=0 corresponds to max_freq and y increases towards min_freq.
        f_hz = float(np.clip(f_hz, self._min_freq_hz, self._max_freq_hz))
        f_bin = int(round(f_hz * self._fft_win_length_s))
        y_in_crop = self._max_freq_bin - f_bin

        crop_height_in = max(1, (self._max_freq_bin - self._min_freq_bin))
        y_scaled = y_in_crop * (self._spec_height_out / float(crop_height_in))
        return int(round(y_scaled))

    def __getitem__(self, idx: int) -> Tuple[object, int, int]:
        row = self._row(idx)
        uq_idx = int(self.uq_idxs[idx])

        audio_path = self._resolve_audio_path(row.audio_path)

        # Load full audio (already resampled to target samplerate by batdetect2)
        audio = bd2_api.load_audio(audio_path, target_samp_rate=self.samplerate_hz)

        # Slice to the annotated call in time (seconds -> samples)
        start_s = max(0.0, row.start_time)
        end_s = max(start_s, row.end_time)
        start_sample = int(round(start_s * self.samplerate_hz))
        end_sample = int(round(end_s * self.samplerate_hz))
        end_sample = min(end_sample, audio.shape[0])
        audio_slice = audio[start_sample:end_sample]

        # Edge case: if the slice is empty, fall back to a tiny zero signal to keep shapes valid.
        if audio_slice.size == 0:
            audio_slice = np.zeros((int(self.samplerate_hz * 0.01),), dtype=np.float32)

        spec = bd2_api.generate_spectrogram(
            audio_slice,
            samp_rate=self.samplerate_hz,
            config=self.spec_params,
            device="cpu"
        )

        # spec shape: [1, 1, H, W]
        spec_2d = spec[0, 0].detach().cpu().numpy()

        # Crop to provided bbox in (time, freq), but using time relative to the slice.
        # Since we sliced audio to [start_time, end_time], the call occupies [0, end-start] in time.
        duration_s = (audio_slice.shape[0] / float(self.samplerate_hz))
        x0 = self._time_to_x(0.0)
        x1 = self._time_to_x(duration_s)
        y0 = self._freq_to_y(row.high_freq)
        y1 = self._freq_to_y(row.low_freq)

        h, w = spec_2d.shape[-2], spec_2d.shape[-1]
        x0 = int(np.clip(x0, 0, max(0, w - 1)))
        x1 = int(np.clip(x1, 0, w))
        y0 = int(np.clip(y0, 0, max(0, h - 1)))
        y1 = int(np.clip(y1, 0, h))

        if x1 - x0 < self._min_bbox_width_px:
            x1 = min(w, x0 + self._min_bbox_width_px)
        if y1 - y0 < self._min_bbox_height_px:
            y1 = min(h, y0 + self._min_bbox_height_px)

        spec_2d = spec_2d[y0:y1, x0:x1]
        img = self._spec_to_pil(spec_2d)

        target = int(row.species_id)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, uq_idx


def subsample_dataset(dataset: BatDetect2, idxs: np.ndarray) -> BatDetect2:
    idxs = np.asarray(idxs, dtype=int)
    dataset.data = [dataset.data[int(i)] for i in idxs.tolist()]
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset


def subsample_classes(dataset: BatDetect2, include_classes=range(2)) -> BatDetect2:
    include = set(int(c) for c in include_classes)
    cls_idxs = [i for i, r in enumerate(dataset.data) if int(r.species_id) in include]
    dataset = subsample_dataset(dataset, np.array(cls_idxs))
    return dataset


def get_train_val_indices(train_dataset: BatDetect2, val_split=0.2):
    all_targets = np.array([int(r.species_id) for r in train_dataset.data])
    train_classes = np.unique(all_targets)

    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(all_targets == cls)[0]
        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_batdetect2_datasets(
    train_transform,
    test_transform,
    train_classes,
    prop_train_labels=0.8,
    split_train_val=False,
    seed=0,
    *,
    csv_path: str,
    audio_root: Optional[str] = None,
):
    """
    Construct BatDetect2 dataset dict in the same format as other SimGCD datasets.
    """
    np.random.seed(seed)

    whole_training_set = BatDetect2(
        csv_path=csv_path,
        audio_root=audio_root,
        train=True,
        transform=train_transform,
    )

    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets (optional; typically unused in this repo)
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    test_dataset = BatDetect2(
        csv_path=csv_path,
        audio_root=audio_root,
        train=False,
        transform=test_transform,
    )

    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    return {
        "train_labelled": train_dataset_labelled,
        "train_unlabelled": train_dataset_unlabelled,
        "val": val_dataset_labelled,
        "test": test_dataset,
    }
