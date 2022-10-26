"""
    File      [ rnn.py ]
    Author    [ Heng-Jui Chang (MIT CSAIL) ]
    Synopsis  [ RNN-based encoder. ]
"""

import torch
from torch import nn


class RNNEncoder(nn.Module):
    """
    RNN-based encoder.
    in_dim [int]: input feature dimension
    hid_dim [int]: hidden feature dimension
    n_layers [int]: number of layers
    module [str]: RNN model type
    dropout [float]: dropout rate
    bidirectional [bool]: bidirectional encoding
    """

    def __init__(
        self, in_dim, hid_dim, n_layers, module="LSTM", dropout=0, bidirectional=True
    ):
        super().__init__()

        # RNN model
        self.rnn = getattr(nn, module)(
            input_size=in_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output dimension
        # Bidirectional makes output size * 2
        self.out_dim = hid_dim * (2 if bidirectional else 1)

    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        """
        Input:
            feat [float tensor]: acoustic feature sequence
            feat_len [long tensor]: feature lengths
        Output:
            out [float tensor]: encoded feature sequence
            out_len [long tensor]: encoded feature lengths
        """

        if not self.training:
            self.rnn.flatten_parameters()

        out, _ = self.rnn(feat)

        return out, feat_len
