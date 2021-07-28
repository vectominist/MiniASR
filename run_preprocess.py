#!/usr/bin/env python3
'''
    File      [ run_preprocess.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Preprocesses corpus. ]
'''

import logging
import argparse
import os
from os.path import join, getsize
import json
from joblib import Parallel, delayed

from miniasr.preprocess.generate_vocab import (
    generate_word_char_vocab, generate_subword_vocab)
from miniasr.utils import base_args, logging_args


def parse_arguments():
    ''' Parses arguments from command line. '''
    parser = argparse.ArgumentParser('Corpus preprocessing.')

    # General & critical arguments
    parser.add_argument('--corpus', '-c', type=str,
                        help='Corpus name.')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to corpus.')
    parser.add_argument('--set', '-s', type=str, nargs='+',
                        help='Which subsets to be processed.')
    parser.add_argument('--out', '-o', type=str, default='data/',
                        help='Output directory.')

    # Vocabulary
    parser.add_argument('--gen-vocab', action='store_true',
                        help='Specify whether to generate vocabulary files.')
    parser.add_argument('--char-vocab-size', type=int, default=-1,
                        help='Character vocabulary size.')
    parser.add_argument('--word-vocab-size', type=int, default=-1,
                        help='Word vocabulary size.')
    parser.add_argument('--subword-vocab-size', type=int, default=-1,
                        help='Subword vocabulary size.')

    # Subword units
    parser.add_argument('--gen-subword', action='store_true',
                        help='Specify whether to generate subword vocabulary.')
    parser.add_argument('--subword-mode', type=str, default='unigram',
                        choices=['unigram', 'bpe'],
                        help='Subword training mode.')
    parser.add_argument('--char-coverage', type=float, default=1.0,
                        help='Character coverage.')

    # Add basic arguments
    parser = base_args(parser)

    args = parser.parse_args()
    logging_args(args)

    return args


def main():
    ''' Main function of preprocessing '''
    args = parse_arguments()

    if args.corpus == 'LibriSpeech':
        from miniasr.preprocess.librispeech import find_data
    else:
        # You may add new methods here.
        raise NotImplementedError(f'Unknown corpus {args.corpus}.')

    logging.info(f'Preprocessing {args.corpus} corpus.')
    logging.info(f'Subsets = {args.set}')

    os.makedirs(args.out, exist_ok=True)
    logging.info(f'Results will be saved to {args.out}')

    # Find all data
    logging.info(f'Reading data from {args.path}')
    data_dict_list = [find_data(join(args.path, s)) for s in args.set]
    data_dict, data_list = {}, []
    for d in data_dict_list:
        data_dict = {**data_dict, **d}
        data_list += [v for k, v in d.items()]
    logging.info(f'Found {len(data_list)} audio files.')

    # Save dictionary of data
    json_path = join(args.out, 'data_dict.json')
    logging.info(f'Saving unsorted data dict to {json_path}')
    with open(json_path, 'w') as fp:
        json.dump(data_dict, fp, indent=4, ensure_ascii=False)

    # Sort data by audio file length
    logging.info('Sorting data by audio file length.')
    file_len = Parallel(n_jobs=args.njobs)(
        delayed(getsize)(d['file']) for d in data_list)
    data_list = [d for d, l in sorted(
        zip(data_list, file_len), reverse=True, key=lambda x: x[1])
        if l > 0 and len(d.get('text', 'dummy')) > 0]

    # Save sorted data list
    json_path = join(args.out, 'data_list_sorted.json')
    logging.info(f'Saving sorted data list to {json_path}')
    with open(json_path, 'w') as fp:
        json.dump(data_list, fp, indent=4, ensure_ascii=False)

    # Generate pure text file for LM training
    if data_list[0].get('text', None):
        logging.info('Generating LM file.')
        text_list = [d['text'] for d in data_list]
        text_path = join(args.out, 'text.txt')
        with open(text_path, 'w') as fp:
            fp.write('\n'.join(text_list))

        # Generate vocabulary files.
        if args.gen_vocab:
            logging.info('Generating vocabularies.')

            if args.char_vocab_size > 0:
                logging.info('Generating characters.')
                generate_word_char_vocab(
                    text_list, join(args.out, 'vocab_char.txt'),
                    args.char_vocab_size, 'char')

            if args.word_vocab_size > 0:
                logging.info('Generating words.')
                generate_word_char_vocab(
                    text_list, join(args.out, 'vocab_word.txt'),
                    args.word_vocab_size, 'word')

            if args.gen_subword and args.subword_vocab_size > 0:
                logging.info('Generating subwords.')
                generate_subword_vocab(
                    text_path,
                    join(args.out,
                         f'{args.subword_mode}_{args.subword_vocab_size}'),
                    vocab_size=args.subword_vocab_size,
                    model_type=args.subword_mode,
                    character_coverage=args.char_coverage)


if __name__ == '__main__':
    main()
