'''
    File      [ dataloader.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ General dataset ]
'''

import logging
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset


class ASRDataset(Dataset):
    '''
        General dataset for ASR
        paths [list]: Paths to preprocessed data dict (.json)
        tokenizer [_BaseTextEncoder]: text tokenizer (see data/tokenizer.py)
    '''

    def __init__(
            self, paths, tokenizer,
            mode='train', paired=True, max_len=1600):
        super().__init__()

        self.paired = paired

        # Load preprocessed dictionaries
        logging.info(f'Loading data from {paths}')
        data_list = []
        for path in paths:
            with open(path, 'r') as fp:
                d_list = json.load(fp)
            data_list += d_list

        self.mode = mode \
            if ((data_list[0].get('text', None) is not None)
                and (tokenizer is not None)) else 'wild'

        if self.mode != 'wild':
            # Tokenize text data
            # Note: 'wild' mode does not have transcription
            for data in tqdm(data_list):
                data['text'] = tokenizer.encode(data['text'])

        self.data_list = [d for d in data_list
                          if len(d.get('text', [0])) > 0]

        self.features = None
        if self.data_list[0]['file'].split('.')[-1] == 'npy':
            # All features are extracted to a single numpy file
            self.features = np.load(self.data_list[0]['file'], mmap_mode='r')
            if mode == 'train':
                self.data_list = [d for d in self.data_list
                                  if d['t_end'] - d['t_begin'] <= max_len]

        logging.info(
            f'{len(self.data_list)} audio files found '
            f'(mode = {self.mode}, paired = {self.paired})')

    def __getitem__(self, index):
        ''' Returns a single sample. '''

        text_index = index if self.paired else np.random.randint(len(self))
        out_dict = {
            'file': self.data_list[index]['file'],
            'vad_segs': self.data_list[index].get('vad_segs', []),
            'text': self.data_list[text_index]['text']
        }

        if self.features is not None:
            out_dict['feat'] = torch.from_numpy(
                self.features[
                    self.data_list[index]['t_begin']:
                    self.data_list[index]['t_end']].copy())
            out_dict['vad_segs'] = []

        return out_dict

    def __len__(self):
        ''' Size of the dataset. '''
        return len(self.data_list)
