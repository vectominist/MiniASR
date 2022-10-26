#!/bin/bash
# Script for getting a sample output from a pre-trained ASR

. ./path.sh || exit 1;

get_sample.py \
    /data/sls/r/u/hengjui/home/scratch/dataset/miniasr_data/timit_phone/dev/data_list_sorted.json \
    model/ctc_conf-cif_timit_phone_7/epoch=129-step=18720.ckpt \
    model/ctc_conf-cif_timit_phone_7/samples \
    -i 0
