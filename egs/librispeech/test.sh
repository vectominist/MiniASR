#!/bin/bash
# Script for ASR training

. ./path.sh || exit 1;

run_asr.py \
    --config config/ctc_test_960.yaml \
    --test \
    --override " \
            args.data.dev_paths=['/work/harry87122/dataset/miniasr_data/test-other/data_list_sorted.json'],, \
            args.decode.type='beam' \
        " \
    --ckpt model/ctc_hubert_base_LS960_char/epoch=15-step=140623.ckpt
