from collections import defaultdict

import torch
from torch import Tensor, nn

from .attention import AttentionBlock, Linear
from .embedding import PositionalEncoding
from .encoder_module import ConvBlock, FeedForwardNet
from .masking import len_to_mask


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        attention_module: str,
        num_heads: int,
        ffn_size: int,
        dropout: float = 0.1,
        activation: str = "relu",
        eps: float = 1e-6,
        norm_first: bool = False,
        **kwargs
    ):
        super().__init__()

        self.attention = AttentionBlock(
            d_model,
            num_heads,
            module=attention_module,
            dropout=dropout,
            eps=eps,
            norm_first=norm_first,
            **kwargs
        )
        self.ffn = FeedForwardNet(
            d_model,
            ffn_size,
            dropout=dropout,
            activation=activation,
            eps=eps,
            norm_first=norm_first,
        )

    def forward(self, x: Tensor, mask: Tensor = None, pos_emb: Tensor = None) -> Tensor:
        sa_res = self.attention(x, mask=mask, pos_emb=pos_emb)
        x = sa_res["x"]
        x = self.ffn(x)

        return x, sa_res


class ConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        attention_module: str,
        num_heads: int,
        ffn_size: int,
        conv_kernel: int = 15,
        dropout: float = 0.1,
        activation: str = "swish",
        eps: float = 1e-6,
        norm_first: bool = False,
        factor: float = 0.5,
        **kwargs
    ):
        super().__init__()

        self.attention = AttentionBlock(
            d_model,
            num_heads,
            module=attention_module,
            dropout=dropout,
            eps=eps,
            norm_first=norm_first,
            **kwargs
        )
        self.conv = ConvBlock(
            d_model, conv_kernel, dropout=dropout, eps=eps, norm_first=norm_first
        )
        self.ffn_1 = FeedForwardNet(
            d_model,
            ffn_size,
            dropout=dropout,
            activation=activation,
            eps=eps,
            norm_first=norm_first,
            factor=factor,
        )
        self.ffn_2 = FeedForwardNet(
            d_model,
            ffn_size,
            dropout=dropout,
            activation=activation,
            eps=eps,
            norm_first=norm_first,
            factor=factor,
        )

        self.final_norm = None
        if norm_first:
            self.final_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: Tensor, mask: Tensor = None, pos_emb: Tensor = None) -> Tensor:
        x = self.ffn_1(x)
        sa_res = self.attention(x, mask=mask, pos_emb=pos_emb)
        x = sa_res["x"]
        x = self.conv(x)
        x = self.ffn_2(x)
        if self.final_norm:
            x = self.final_norm(x)

        return x, sa_res


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        num_layers: int,
        module: str = "transformer",
        attention_module: str = "pytorch",
        num_heads: int = 8,
        ffn_size: int = 1024,
        dropout: float = 0.1,
        eps: float = 1e-6,
        norm_first: bool = False,
        max_len: int = 3600,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.attention_module = attention_module
        self.module = module
        self.out_dim = d_model

        self.pre_proj = Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        if module == "transformer":
            self.encoder_layers = nn.ModuleList(
                [
                    TransformerEncoderLayer(
                        d_model,
                        attention_module,
                        num_heads,
                        ffn_size,
                        dropout=dropout,
                        eps=eps,
                        norm_first=norm_first,
                        **kwargs
                    )
                    for _ in range(num_layers)
                ]
            )
        if module == "conformer":
            self.encoder_layers = nn.ModuleList(
                [
                    ConformerEncoderLayer(
                        d_model,
                        attention_module,
                        num_heads,
                        ffn_size,
                        dropout=dropout,
                        eps=eps,
                        norm_first=norm_first,
                        **kwargs
                    )
                    for _ in range(num_layers)
                ]
            )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        x_len: Tensor,
        get_hidden: bool = False,
        get_attention: bool = False,
    ):
        B, T, _ = x.shape
        pad_mask = len_to_mask(x_len)
        pos_emb = self.pos_enc(T)
        x = self.pre_proj(x)

        if self.attention_module in {"pytorch", "vq"}:
            x = self.dropout(x) + pos_emb

        other_res = defaultdict(list)
        for i in range(len(self.encoder_layers)):
            x, sa_res = self.encoder_layers[i](x, pad_mask, pos_emb)
            if get_hidden:
                other_res["hidden"].append(x)
            if get_attention:
                other_res["attn"].append(sa_res["attn"])
            if self.attention_module == "vq":
                other_res["kmeans_loss"].append(sa_res["kmeans_loss"])
                other_res["code_perplexity"].append(sa_res["code_perplexity"])

        return x, other_res
