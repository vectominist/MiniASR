#!/bin/bash
# Script for getting a sample output from a pre-trained ASR

. ./path.sh || exit 1;

name=ctc_conf-cif-inn-w5_timit_phone_1

get_sample.py \
    /data/sls/r/u/hengjui/home/scratch/dataset/miniasr_data/timit_phone/dev/data_list_sorted.json \
    model/$name/epoch=304-step=43920.ckpt \
    model/$name/samples \
    -i 0
