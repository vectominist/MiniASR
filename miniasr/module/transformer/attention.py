# ref: https://github.com/sooftware/conformer

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoder_module import Linear
from .vq_attention import VectorQuantizedSelfAttention


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = (
            self.key_proj(key)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        value = (
            self.value_proj(value)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        pos_embedding = self.pos_proj(pos_embedding).view(
            batch_size, -1, self.num_heads, self.d_head
        )

        content_score = torch.matmul(
            (query + self.u_bias).transpose(1, 2), key.transpose(2, 3)
        )
        pos_score = torch.matmul(
            (query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1)
        )
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, float("-inf"))

        attn_raw = F.softmax(score, -1)
        attn = self.dropout(attn_raw)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context), attn_raw

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(
            batch_size, num_heads, seq_length2 + 1, seq_length1
        )
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        module: str = "pytorch",
        dropout: float = 0.1,
        eps: float = 1e-6,
        norm_first: bool = True,
        **kwargs
    ):
        super().__init__()
        # self.positional_encoding = PositionalEncoding(d_model)
        self.module = module
        if module == "pytorch":
            self.attention = nn.MultiheadAttention(
                d_model, num_heads, batch_first=True, dropout=dropout
            )
        if module == "relative":
            self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout)
        if module == "vq":
            self.attention = VectorQuantizedSelfAttention(
                d_model, num_heads, dropout=dropout, **kwargs
            )

        self.layernorm = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_first = norm_first

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None, pos_emb: Optional[Tensor] = None
    ):
        res = x
        if self.norm_first:
            x = self.layernorm(x)
        if self.module == "pytorch":
            x, attn = self.attention(
                x, x, x, key_padding_mask=~mask, average_attn_weights=False
            )
        if self.module == "relative":
            x, attn = self.attention(
                x, x, x, pos_embedding=pos_emb, mask=~mask.unsqueeze(1)
            )
        if self.module == "vq":
            result = self.attention(x, ~mask)
            x = result["out"]
            attn = result["attn"]

        x = self.dropout(x)
        x = x + res
        if not self.norm_first:
            x = self.layernorm(x)

        outputs = {"x": x, "attn": attn}
        if self.module == "vq":
            outputs["kmeans_loss"] = result["kmeans_loss"]
            outputs["code_perplexity"] = result["code_perplexity"]
            outputs["targets"] = result["targets"]

        return outputs
