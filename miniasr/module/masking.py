'''
    File      [ masking.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Masking-related. ]
'''

import torch


def len_to_mask(lengths, max_length=-1, dtype=None):
    '''
        Converts lengths to a binary mask.
            lengths [long tensor]
        E.g.
            lengths = [5, 3, 1]
            mask = [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0]
            ]
    '''
    max_length = max_length if max_length > 0 else lengths.max().cpu().item()
    mask = (torch.arange(max_length, device=lengths.device, dtype=lengths.dtype)
            .expand(lengths.shape[0], max_length) < lengths.unsqueeze(1))
    if dtype is not None:
        mask = mask.type(dtype)
    return mask
