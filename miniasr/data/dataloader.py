'''
    File      [ dataloader.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ General data loader. ]
'''

import logging
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from miniasr.data.dataset import ASRDataset
from miniasr.data.text import load_text_encoder
from miniasr.data.audio import load_waveform

TARGET_SR = 16_000  # target audio sample rate: 16kHz


def audio_collate_fn(data_list, mode='train'):
    '''
        Collate function for ASR
        Input:
            data_list [list of dicts]: list of items from dataset
            mode [str]: 'train' has paired audio and text data
                        'wild' has audio data only
        Output: (dict)
            wave [list of tensors]: raw waveforms
            wave_len [long tensor]: lengths of waveforms
            text [long tensor]: encoded text sequences
            text_len [long tensor]: lengths of text sequences
    '''

    waves, wave_len = [], []
    texts, text_len = [], []
    for data in data_list:
        # Load raw waveform
        waves.append(load_waveform(data['file'], TARGET_SR))
        wave_len.append(len(waves[-1]))

        if mode in ['train', 'dev']:
            texts.append(torch.LongTensor(data['text']))
            text_len.append(len(texts[-1]))
        else:
            texts.append(data.get('text', ''))
            text_len.append(0)

    wave_len = torch.LongTensor(wave_len)

    if mode in ['train', 'dev']:
        texts = pad_sequence(texts, batch_first=True)
        text_len = torch.LongTensor(text_len)

    return {'file': [data['file'] for data in data_list],
            'wave': waves, 'wave_len': wave_len,
            'text': texts, 'text_len': text_len}


def text_collate_fn(data_list):
    '''
        Collate function for text data only
        Input:
            data [list of dicts]: list of items from dataset
        Output: (dict)
            text [long tensor]: encoded text sequences
            text_len [long tensor]: lengths of text sequences
    '''

    texts, text_len = [], []
    for data in data_list:
        texts.append(torch.LongTensor(data['text']))
        text_len.append(len(texts[-1]))

    texts = pad_sequence(texts, batch_first=True)
    text_len = torch.LongTensor(text_len)

    return {'text': texts, 'text_len': text_len}


def create_dataloader(args, tokenizer=None):
    ''' Creates datasets and dataloaders '''

    if tokenizer is None:
        # Create text tokenizer
        logging.info(
            f'Creating text tokenizer of {args.data.text.mode} level.')
        tokenizer = load_text_encoder(
            args.data.text.mode, args.data.text.vocab)

    # Create datasets & dataloaders
    logging.info(f'Generating datasets and dataloaders. (mode = {args.mode})')
    if args.mode == 'train':
        # Training mode: train + dev sets
        tr_set = ASRDataset(args.data.train_paths, tokenizer, 'train')
        dv_set = ASRDataset(args.data.dev_paths, tokenizer, 'dev')

        tr_loader = DataLoader(
            tr_set, batch_size=args.hparam.train_batch_size,
            shuffle=True, num_workers=args.hparam.njobs,
            collate_fn=audio_collate_fn,
            pin_memory=args.hparam.pin_memory,
            drop_last=True)
        dv_loader = DataLoader(
            dv_set, batch_size=args.hparam.val_batch_size,
            shuffle=False, num_workers=args.hparam.njobs,
            collate_fn=audio_collate_fn,
            pin_memory=args.hparam.pin_memory,
            drop_last=False)

        return tr_loader, dv_loader, tokenizer
    elif args.mode in ['dev', 'wild']:
        # Dev mode: dev set only (w/ transcriptions)
        # Wild mode: w/o transcriptions
        dv_set = ASRDataset(args.data.dev_paths, tokenizer, args.mode)

        dv_loader = DataLoader(
            dv_set, batch_size=args.hparam.val_batch_size,
            shuffle=False, num_workers=args.hparam.njobs,
            collate_fn=partial(audio_collate_fn, mode=args.mode),
            pin_memory=args.hparam.pin_memory,
            drop_last=False)

        return None, dv_loader, tokenizer
    else:
        raise NotImplementedError(f'Unknown mode {args.mode}')
