'''
    File      [ encoder_transformer.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Transformer-based encoder. ]
'''

import math
import torch
from torch import nn

from miniasr.module.masking import len_to_mask


class PositionalEncoding(nn.Module):
    '''
        Positional Encoding
    '''

    def __init__(self, d_emb, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_emb, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_emb, 2)
                             * (-math.log(10000.0) / d_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, seq):
        ''' Add PE to input. '''
        output = seq + self.pe[:, :seq.shape[1]]
        return self.dropout(output)


class TransformerEncoder(nn.Module):
    '''
        Transformer-based encoder.
        in_dim [int]: input feature dimension
        hid_dim [int]: hidden feature dimension
        n_layers [int]: number of layers
        module [str]: RNN model type
        dropout [float]: dropout rate
        bidirectional [bool]: bidirectional encoding
    '''

    def __init__(self, in_dim, hid_dim, n_layers, n_heads,
                 dim_feedforward, dropout=0, activation='relu'):
        super().__init__()

        # Model
        self.in_layer = nn.Linear(in_dim, hid_dim)
        self.pos_enc = PositionalEncoding(hid_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            hid_dim, n_heads, dim_feedforward, dropout, activation)
        self.model = nn.TransformerEncoder(
            layer, n_layers, nn.LayerNorm(hid_dim, 1e-12))

        # Output dimension
        self.out_dim = hid_dim

    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        '''
            Input:
                feat [float tensor]: acoustic feature sequence
                feat_len [long tensor]: feature lengths
            Output:
                out [float tensor]: encoded feature sequence
                out_len [long tensor]: encoded feature lengths
        '''

        feat = self.in_layer(feat)
        feat = self.pos_enc(feat)
        pad_mask = ~len_to_mask(feat_len, dtype=torch.bool)
        out = self.model(
            feat.transpose(0, 1), src_key_padding_mask=pad_mask)

        return out.transpose(0, 1), feat_len
