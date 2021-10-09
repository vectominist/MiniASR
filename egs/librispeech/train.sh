#!/bin/bash
# Script for ASR training

. ./path.sh || exit 1;

mkdir -p model

run_asr.py \
    --config config/ctc_train_960h.yaml
    # --ckpt 'path/to/ckpt'
