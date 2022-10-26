import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .activation import GLU, Swish, get_activation

# ref: https://github.com/sooftware/conformer


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class FeedForwardNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        eps: float = 1e-6,
        norm_first: bool = False,
        factor: float = 1.0,
    ):
        super().__init__()

        self.fc_1 = nn.Linear(d_model, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, d_model)
        self.layernorm = nn.LayerNorm(d_model, eps=eps)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.factor = factor

    def forward(self, x: Tensor) -> Tensor:
        res = x
        if self.norm_first:
            x = self.layernorm(x)
        x = self.fc_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = self.dropout(x) * self.factor
        x = res + x
        if not self.norm_first:
            x = self.layernorm(x)

        return x


class DepthwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
    ) -> None:
        super().__init__()

        self.padding = padding
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dropout: float = 0.1,
        eps: float = 1e-6,
        norm_first: bool = True,
    ):
        super().__init__()

        self.conv_1 = nn.Conv1d(d_model, d_model * 2, 1)
        self.conv_2 = DepthwiseConv1d(
            d_model, d_model, kernel_size, padding=calc_same_padding(kernel_size)
        )
        self.conv_3 = nn.Conv1d(d_model, d_model, 1)

        self.layernorm = nn.LayerNorm(d_model, eps=eps)
        self.glu = GLU(dim=1)
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        res = x
        if self.norm_first:
            x = self.layernorm(x)
        x = self.conv_1(x.transpose(1, 2))  # (B, 2D, T)
        x = self.glu(x)  # (B, D, T)
        x = self.conv_2(x)
        x = self.batchnorm(x)
        x = self.swish(x)
        x = self.conv_3(x)
        x = self.dropout(x).transpose(1, 2)
        x = x + res
        if not self.norm_first:
            x = self.layernorm(x)

        return x
