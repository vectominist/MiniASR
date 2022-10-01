#!/bin/bash
# Script for generating librispeech data files

# . ./path.sh || exit 1;

corpus=TIMIT
corpus_dir=/data/sls/d/corpora/original/timit
out_dir=/data/sls/scratch/hengjui/dataset/miniasr_data
vocab_type=text

mkdir -p $out_dir

for set in train
do
    echo "Preprocessing $set set of $corpus."
    run_preprocess.py \
        -c $corpus \
        -p $corpus_dir \
        -s $set \
        -o $out_dir/$set \
        --gen-vocab \
        --source $vocab_type \
        --char-vocab-size 40 \
        --word-vocab-size 10000 \
        --njobs 16
done

for set in test
do
    echo "Preprocessing $set set of $corpus."
    run_preprocess.py \
        -c $corpus \
        -p $corpus_dir \
        -s $set \
        -o $out_dir/$set \
        --njobs 16
done
