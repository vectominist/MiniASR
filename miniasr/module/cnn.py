"""
    File      [ cnn.py ]
    Author    [ Heng-Jui Chang (MIT CSAIL) ]
    Synopsis  [ RNN-based encoder. ]
"""

from typing import List, Tuple

import torch
from torch import LongTensor, Tensor, nn
from torch.nn.utils.rnn import pad_sequence


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


class DownsampleConv2dGT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        downsample: float = 4.0,
    ) -> None:
        super().__init__()

        self.downsample = downsample
        self.conv = SameConv2d(in_dim, hid_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: Tensor, x_len: LongTensor, boundaries: List[List[int]]):
        # x: (B, T, D)
        # x_len: (B, )

        # x = self.conv(x)
        x_len_d = x_len.cpu().tolist()

        out_list = []
        out_len_list = []
        for b in range(x.shape[0]):
            out = []
            prev_t = 0
            if self.downsample > 1.0:
                rate = x_len_d[b] / len(boundaries[b]) / self.downsample
            if boundaries[b][-1] < x_len_d[b]:
                boundaries[b].append(x_len_d[b])
            for t in boundaries[b]:
                t = min(t, x_len_d[b])
                if prev_t == t:
                    continue
                if self.downsample > 1.0:
                    delta_t = int(round((t - prev_t) / rate))
                    if delta_t <= 0:
                        out.append(x[b, prev_t:t].mean(0))
                    else:
                        _t = prev_t
                        while _t < t:
                            out.append(x[b, _t : min(_t + delta_t, t)].mean(0))
                            _t += delta_t
                else:
                    out.append(x[b, prev_t:t].mean(0))
                prev_t = t

            out = torch.stack(out, dim=0)
            out_list.append(out)
            out_len_list.append(len(out))

        out = pad_sequence(out_list, batch_first=True)
        out_len = LongTensor(out_len_list).to(x_len.device)

        out = self.conv(out)

        return out, out_len
