
'''
    File      [ encoder_rnn.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ RNN-based encoder. ]
'''

import torch
from torch import nn
from fairseq.modules import SamePad


class CNNEncoder(nn.Module):
    '''
        CNN-based encoder.
        in_dim [int]: input feature dimension
        hid_dim [int]: hidden feature dimension
        n_layers [int]: number of layers
        kernel_size [int]: kernel size
        stride [int]: stride
        dropout [float]: dropout rate
        batch_norm [bool]: enables batch normalization
        causal [bool]: causal convolution
        activation [str]: activation function (in torch.nn)
    '''

    def __init__(self, in_dim, hid_dim, out_dim=None,
                 kernel_size=[3, 3], stride=[1, 1],
                 dropout=0, batch_norm=False, causal=False,
                 activation='ReLU'):
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = hid_dim if out_dim is None else out_dim

        self.n_layers = len(kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_norm = batch_norm
        self.causal = causal

        modules = []
        self.padding, self.same_pad_size = [], []
        for i, (k_size, s_size) in enumerate(zip(kernel_size, stride)):
            padding = k_size - 1 if causal else k_size // 2
            modules += [
                nn.Conv1d(
                    in_dim if i == 0 else hid_dim,
                    hid_dim if i < self.n_layers - 1 else self.out_dim,
                    kernel_size=k_size,
                    stride=s_size,
                    padding=padding),
                SamePad(k_size, causal=causal)
            ]
            self.padding.append(padding)
            self.same_pad_size.append(modules[-1].remove)
            if i < self.n_layers - 1:
                if batch_norm:
                    modules.append(nn.BatchNorm1d(hid_dim))
                modules += [getattr(nn, activation)(), nn.Dropout(dropout)]

        self.layers = nn.ModuleList(modules)

    def compute_out_len(self, feat_len):
        ''' Computes output lengths. '''

        if feat_len is None:
            return None

        for k_size, s_size, p_size, r_size in zip(
                self.kernel_size, self.stride, self.padding, self.same_pad_size):
            feat_len = (feat_len + 2 * p_size - k_size) // s_size + 1 - r_size

        return feat_len

    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        '''
            Input:
                feat [float tensor]: acoustic feature sequence
                feat_len [long tensor]: feature lengths
            Output:
                out [float tensor]: encoded feature sequence
        '''

        feat = feat.transpose(1, 2)  # (Batch x Dim x Time)

        for layer in self.layers:
            feat = layer(feat)

        feat = feat.transpose(1, 2)  # (Batch x Time x Dim)

        return feat, self.compute_out_len(feat_len)
