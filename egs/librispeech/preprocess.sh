#!/bin/bash
# Script for generating librispeech data files

. ./path.sh || exit 1;

corpus=LibriSpeech
corpus_dir=/work/harry87122/dataset/LibriSpeech
# corpus_dir=/work/harry87122/dataset/librispeech_finetuning
out_dir=/work/harry87122/dataset/miniasr_data

mkdir -p $out_dir

for set in train-clean-100 train-clean-360 train-other-500
do
    echo "Preprocessing $set set of $corpus."
    run_preprocess.py \
        -c $corpus \
        -p $corpus_dir \
        -s $set \
        -o $out_dir/$set \
        --gen-vocab \
        --char-vocab-size 40 \
        --word-vocab-size 10000 \
        --subword-vocab-size 5000 \
        --gen-subword \
        --subword-mode unigram \
        --char-coverage 1.0 \
        --njobs 16
done

for set in dev-clean test-clean dev-other test-other
do
    echo "Preprocessing $set set of $corpus."
    run_preprocess.py \
        -c $corpus \
        -p $corpus_dir \
        -s $set \
        -o $out_dir/$set \
        --njobs 16
done
