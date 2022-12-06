#!/bin/bash
# Script for ASR training

. ./path.sh || exit 1;

mkdir -p model

run_asr.py \
    --config config/ctc_train_transformer.yaml \
    --detect-anomaly
    # --ckpt 'path/to/ckpt'
