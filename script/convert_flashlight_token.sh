#!/bin/bash
# Converts vocab file to flashlight format.

vocab_file=$1
out_file=$2

cp $vocab_file $out_file

sed -i 's/ /|/g' $out_file
sed -i '1s/^/#\n<s\/>\n<unk>\n/' $out_file
