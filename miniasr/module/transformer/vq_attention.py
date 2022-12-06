import math
from typing import Any, Dict, Tuple

import torch

# import torch.nn.functional as F
from torch import Tensor, nn

from .encoder_module import Linear
from .kmeans_vector_quantizer import KmeansVectorQuantizer


def show_val(tag: str, x: Tensor):
    return
    with torch.no_grad():
        print(tag + ":", x.norm(), x.mean(), x.min(), x.max())


class VectorQuantizedSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        codebook_size: int = 64,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.codebook_size = codebook_size

        self.vq = KmeansVectorQuantizer(
            dim=d_model,
            num_vars=codebook_size,
            groups=num_heads,
            combine_groups=True,
            vq_dim=d_model,
            time_first=True,
        )
        # self.vq = Linear(d_model, num_heads * codebook_size)
        self.proj_q = Linear(self.vq.var_dim, self.vq.var_dim)
        self.proj_k = Linear(self.vq.var_dim, self.vq.var_dim)
        self.proj_v = Linear(d_model, d_model // num_heads)
        self.norm_factor = 1 / math.sqrt(self.vq.var_dim)

        # Similarity matrix: (G, V, V)
        # self.sim = nn.Parameter(
        #     10 * torch.randn(num_heads, codebook_size, codebook_size)
        # )
        self.temp = nn.Parameter(torch.tensor(-math.log(0.07)))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # x: (B, T, D)
        B, T, D = x.shape

        vq_res = self.vq(x, produce_targets=True)
        x_vq = vq_res["x"]  # (B, T, D)
        d = vq_res["distance"]  # (B, T, G, V)
        kmeans_loss = vq_res["kmeans_loss"]
        d = -d * torch.exp(self.temp)

        # d = self.vq(x).reshape(B, T, self.num_heads, self.codebook_size)

        prob_v = torch.softmax(d, dim=-1)  # (B, T, G, V)
        show_val("prob_v", prob_v)

        # kmeans_loss = 1.0 - (-prob_v * torch.log(prob_v + 1e-9)).sum(-1).exp().mean()

        if mask is not None:
            # prob_v.masked_fill_(mask.view(B, T, 1, 1), 0)
            d.masked_fill_(mask.view(B, T, 1, 1), float("-inf"))
        prob_t = torch.softmax(d, dim=1)  # (B, T, G, V)
        show_val("prob_t", prob_t)

        # d_exp = torch.exp(-d * torch.exp(self.temp))  # (B, T, G, V)
        # d_exp = d_exp * (~pad_mask.view(B, T, 1, 1)).type(d_exp.dtype)
        # prob_t = d_exp / d_exp.sum(1, keepdim=True)  # (B, T, G, V)

        # e_new = (prob_t.unsqueeze(-1) * x_vq.view(B, T, 1, 1, D)).sum(1)
        e_new = (prob_t.unsqueeze(-1) * x.view(B, T, 1, 1, D)).sum(1)
        value = self.proj_v(e_new)  # (B, G, V, D/G)
        value = self.dropout(value)
        # e_new: (B, G, V, D)
        # TODO: average over batch? e_new -> (G, V, D)
        show_val("value", value)

        # prob_v = d_exp / d_exp.sum(-1, keepdim=True)  # (B, T, G, V)

        # score = torch.einsum("btgv,gvu->btgu", prob_v, self.sim)  # (B, T, G, V)
        # self.vq.embedding: (V, G, D)
        q = self.proj_q(self.vq.embedding).transpose(0, 1)  # (G, V, D)
        k = self.proj_k(self.vq.embedding).permute(1, 2, 0)  # (G, D, V)
        sim = torch.bmm(q, k) * self.norm_factor  # (G, V, V)
        sim = sim.expand(self.num_heads, -1, -1)
        score = (
            torch.bmm(
                prob_v.permute(2, 0, 1, 3).reshape(
                    self.num_heads, B * T, self.codebook_size
                ),
                sim,
            )
            .reshape(self.num_heads, B, T, self.codebook_size)
            .permute(1, 2, 0, 3)
        )
        show_val("score", score)
        attn = torch.softmax(score, dim=-1)  # (B, T, G, V)
        show_val("attn", attn)
        attn_drop = self.dropout(attn)
        out = (
            (attn_drop.unsqueeze(-1) * value.unsqueeze(1)).sum(3).reshape(B, T, D)
        )  # (B, T, D)

        assert out.shape == x.shape, (out.shape, x.shape)

        show_val("out", out)

        return {
            "out": out,
            "attn": attn,
            "kmeans_loss": kmeans_loss,
            "code_perplexity": vq_res["code_perplexity"],
            "targets": vq_res["targets"],
            "similarity": sim,
        }
