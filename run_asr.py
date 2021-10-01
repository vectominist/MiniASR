#!/usr/bin/env python3
'''
    File      [ run_asr.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ End-to-end ASR training. ]
'''

import logging
import argparse
import json
import os
from os.path import join
import yaml
from easydict import EasyDict as edict
import torch
import torchaudio

from miniasr.bin.asr_trainer import create_asr_trainer, create_asr_trainer_test
from miniasr.utils import set_random_seed, base_args, logging_args, override


def parse_arguments():
    ''' Parses arguments from command line. '''
    parser = argparse.ArgumentParser('End-to-end ASR training.')

    parser.add_argument('--config', '-c', type=str, default='none',
                        help='Training configuration file (.yaml).')

    # Testing
    parser.add_argument('--test', '-t', action='store_true',
                        help='Specify testing mode.')
    parser.add_argument('--ckpt', type=str, default='none',
                        help='Checkpoint for testing.')
    parser.add_argument('--test-name', type=str, default='test_result',
                        help='Specify testing results\' name.')

    parser = base_args(parser)  # Add basic arguments
    args = parser.parse_args()
    logging_args(args)  # Set logging

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # Load config file.
    if args.config != 'none':
        config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    else:
        config = {}
        if args.ckpt != 'none':
            logging.info(
                f'Did not provide config file, using {args.ckpt} instead')
        else:
            raise RuntimeError('No config file and ckpt found!')

    args = edict({**config, **vars(args)})

    if args.cpu:
        args.trainer.gpus = 0

    if args.override != '':
        override(args.override, args)

    return args


def main():
    ''' Main function of ASR training. '''

    # Basic setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    torchaudio.set_audio_backend('sox_io')

    # Parse arguments
    args = parse_arguments()

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Create base directory
    if args.test:
        assert args.ckpt != 'none'
        # Path to save testing results.
        args.test_res = '/'.join(args.ckpt.split('/')[:-1])

    # Save a copy of args
    if args.config != 'none':
        os.makedirs(args.trainer.default_root_dir, exist_ok=True)
        args_path = join(args.trainer.default_root_dir,
                         f'model_{args.mode}_config.yaml')
        with open(args_path, 'w') as fp:
            yaml.dump(json.loads(json.dumps(args)), fp,
                      indent=2, encoding='utf-8')

    # Get device
    device = torch.device('cpu' if args.cpu else 'cuda:0')

    if not args.test:
        # Training
        logging.info('Training mode.')
        args, tr_loader, dv_loader, _, model, trainer = \
            create_asr_trainer(args, device)
        trainer.fit(model, tr_loader, dv_loader)
    else:
        # Testing
        logging.info('Testing mode.')
        args, _, dv_loader, _, model, trainer = \
            create_asr_trainer_test(args, device)
        model.eval()
        trainer.test(model, dv_loader)


if __name__ == '__main__':
    main()
