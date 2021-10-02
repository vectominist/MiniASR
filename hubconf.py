'''
    File      [ hubconf.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Hubconf. ]
'''

import os
import torch
from miniasr.utils import load_from_checkpoint


def asr_local(ckpt):
    '''
        ASR model from a local checkpoint.
    '''
    assert os.path.isfile(ckpt)
    model, _, _ = load_from_checkpoint(ckpt, 'cpu')
    return model


def asr_url(ckpt):
    '''
        ASR model from an url.
    '''
    ckpt = torch.hub.load_state_dict_from_url(ckpt)
    model, _, _ = load_from_checkpoint(ckpt, 'cpu')
    return model


def ctc_eng():
    '''
        Default Englsih CTC model.
    '''
    return ctc_eng_ls960_hubert_base_char()


def ctc_eng_ls960_hubert_base_char():
    '''
        Language: English
        Data: LibriSpeech 960h
        Feature: HuBERT base
        Vocab: char
    '''
    return asr_url('https://www.dropbox.com/s/1k3mpngqpinihlo/ctc_ls960_hubert_base_char.ckpt?dl=0')
