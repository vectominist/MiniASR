#!/bin/bash
# Script for ASR training

. ./path.sh || exit 1;

name=$1

run_asr.py \
    --config egs/librispeech/config/ctc_test_100h.yaml \
    --test \
    --test-name $name \
    --ckpt /work/harry87122/MiniASR/model/ctc_base_hubert_LS100_char/epoch=43-step=39247.ckpt
