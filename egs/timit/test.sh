#!/bin/bash
# Script for ASR testing

. ./path.sh || exit 1;

run_asr.py \
    --config config/ctc_test_rnn.yaml \
    --test \
    --override " \
        args.data.dev_paths=['/data/sls/r/u/hengjui/home/scratch/dataset/miniasr_data/timit_phone/test/data_list_sorted.json']" \
    --ckpt model/ctc_conf-cif_timit_phone_9/epoch=129-step=18720.ckpt
