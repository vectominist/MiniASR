#!/usr/bin/env python3
'''
    File      [ run_extract.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Extract features with pre-trained models. ]
'''

import warnings
import logging
import argparse
import os
from os.path import join
import json
from functools import partial
from tqdm import tqdm

from npy_append_array import NpyAppendArray
import torch
from torch.utils.data import DataLoader
import torchaudio

from miniasr.data.dataset import ASRDataset
from miniasr.data.dataloader import audio_collate_fn
from miniasr.utils import set_random_seed, base_args, logging_args
from miniasr.model.base_asr import get_model_stride, extracted_length

warnings.filterwarnings("ignore")


def extract_features(
        extractor, feat_select, device, dataloader,
        out_dir, stride, save_type='pt'):
    ''' Extract features with a pre-trained extractor '''
    data_list = []
    feat_dim = 0

    if save_type == 'npy':
        output_path = join(out_dir, 'data.npy')
        if os.path.exists(output_path):
            os.remove(output_path)
        logging.info(
            f'Results will be saved to a single .npy file {output_path}')
        npaa = NpyAppendArray(output_path)
        count = 0

    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            # Get info
            files = data['file']
            waves = data['wave']
            waves_len = data['wave_len']
            texts = data['text']

            # Extract features
            feats = extractor([wave.to(device)
                               for wave in waves])  # B x Time x Dim
            feats = feats[feat_select]
            feats_len = [extracted_length(length, stride)
                         for length in waves_len]

            # Save features
            for file, feat, length, text in zip(files, feats, feats_len, texts):
                res = {'wave_path': file}
                if save_type == 'pt':
                    # Note: LibriSpeech files are .flac
                    speaker_id = file.split('/')[-1].split('-')[0]
                    os.makedirs(join(out_dir, speaker_id), exist_ok=True)
                    output_path = join(
                        out_dir, speaker_id, file.split('/')[-1][:-4] + 'pt')
                    torch.save(feat[:length].cpu(), output_path)
                elif save_type == 'npy':
                    npaa.append(feat[:length].cpu().numpy())
                    res['t_begin'] = count
                    res['t_end'] = count + length.cpu().item()
                    count += length.cpu().item()

                res['file'] = output_path
                res['text'] = text
                data_list.append(res)

            if i == 0:
                feat_dim = feats[0].shape[-1]

    return data_list, feat_dim


def parse_arguments():
    ''' Parses arguments from command line. '''

    parser = argparse.ArgumentParser(
        'Extract features from pre-trained models.')

    # Main variables
    parser.add_argument('--data', '-d', type=str, nargs='+',
                        help='Corpus to extract (LibriSpeech).')
    parser.add_argument('--model', '-m', type=str,
                        help='Pre-trained model.')
    parser.add_argument('--out', '-o', type=str, default='feat/',
                        help='Directory for saving results.')
    parser.add_argument('--save-type', '-s', type=str, default='pt',
                        choices=['pt', 'npy'],
                        help='Saved data type.')

    # Feature-related
    parser.add_argument('--feature', '-f', type=str, default='last_hidden_state',
                        help='Feature selection.')
    parser.add_argument('--batch', '-b', type=int, default=8,
                        help='Batch size.')

    parser = base_args(parser)  # Add basic arguments
    args = parser.parse_args()
    logging_args(args)  # Set logging

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

    # Check device
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda:0')
        torch.cuda.manual_seed_all(args.seed)
        logging.info(f'{torch.cuda.device_count()} GPU(s) available')
        logging.info('Using GPU for faster extraction')
    else:
        device = torch.device('cpu')
        logging.info('Using CPU (might be slower)')

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    logging.info(f'Results will be saved to {args.out}')

    # Load data
    print('Loading data from {}'.format(args.data))
    dataset = ASRDataset(args.data, None)
    dataloader = DataLoader(
        dataset, batch_size=args.batch,
        shuffle=False, num_workers=args.njobs,
        collate_fn=partial(audio_collate_fn, mode='wild'),
        pin_memory=True, drop_last=False)

    # Load pre-trained model
    logging.info(f'Loading pre-trained extractor : {args.model}')
    extractor = torch.hub.load('s3prl/s3prl', args.model).to(device)
    extractor.eval()

    # Extract embeddings
    logging.info('Start extracting features')
    data_list, _ = extract_features(
        extractor, args.feature, device, dataloader, args.out,
        get_model_stride(args.model), args.save_type)

    # Save record dict
    logging.info('Saving results')
    with open(join(args.out, 'feat_list_sorted.json'), 'w') as fp:
        json.dump(data_list, fp, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
