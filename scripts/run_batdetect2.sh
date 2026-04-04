#!/bin/bash

set -e
set -x

# Echolocation CSVs use audio_path like datasets/<source>/audio/*.wav — that is relative to this root
# (NOT .../echolocation/audio). Wrong root → FileNotFoundError for datasets/bcireland/...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMGCD="$(cd "$SCRIPT_DIR/.." && pwd)"
BIOCAI="$(cd "$SIMGCD/../../.." && pwd)"
BATDETECT2_AUDIO_ROOT="${BATDETECT2_AUDIO_ROOT:-$BIOCAI/ch2/datasets/echolocation/audio}"
BATDETECT2_CSV="${BATDETECT2_CSV:-$BIOCAI/ch2/datasets/echolocation/annotations/batdetect2_echospfull.csv}"

CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --dataset_name 'batdetect2' \
    --batdetect2_csv_path "$BATDETECT2_CSV" \
    --batdetect2_audio_root "$BATDETECT2_AUDIO_ROOT" \
    --batch_size 1024 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 16 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.8 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name batdetect2_simgcd

    