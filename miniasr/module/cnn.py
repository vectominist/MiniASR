"""
    File      [ cnn.py ]
    Author    [ Heng-Jui Chang (MIT CSAIL) ]
    Synopsis  [ RNN-based encoder. ]
"""

from typing import Tuple

import torch
from torch import nn


class DownsampleConv2d(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super().__init__()

        self.out_dim = out_dim
        if in_dim == 240:
            self.in_dim = 80
            self.in_channel = 3
        else:
            self.in_dim = in_dim
            self.in_channel = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, hid_dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(hid_dim, hid_dim, 3, 2),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hid_dim * (((self.in_dim - 1) // 2 - 1) // 2), out_dim)

    def forward(
        self, x: torch.Tensor, x_len: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        B, T, D = x.shape
        if self.in_channel == 1:
            x = x.unsqueeze(1)
            # B x 1 x T x D
        else:
            x = x.reshape(B, T, self.in_dim, self.in_channel).permute(0, 3, 1, 2)
            # B x in_channel x T x in_dim

        x = self.conv(x)
        # B x hid_dim x T//4 x in_dim//4
        B, H, T, D = x.shape
        x = self.proj(x.permute(0, 2, 3, 1).reshape(B, T, D * H))
        # x_len = ((x_len - 1) // 2 - 1) // 2
        x_len = torch.div(x_len - 1, 2, rounding_mode="floor")
        x_len = torch.div(x_len - 1, 2, rounding_mode="floor")

        return x, x_len


class SameConv2d(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super().__init__()

        self.out_dim = out_dim
        if in_dim == 240:
            self.in_dim = 80
            self.in_channel = 3
        else:
            self.in_dim = in_dim
            self.in_channel = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, hid_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hid_dim, hid_dim, 3, 1, 1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hid_dim * self.in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        B, T, D = x.shape
        if self.in_channel == 1:
            x = x.unsqueeze(1)
            # B x 1 x T x D
        else:
            x = x.reshape(B, T, self.in_dim, self.in_channel).permute(0, 3, 1, 2)
            # B x in_channel x T x in_dim

        x = self.conv(x)
        # B x hid_dim x T x in_dim
        B, H, T, D = x.shape
        x = self.proj(x.permute(0, 2, 3, 1).reshape(B, T, D * H))

        return x
