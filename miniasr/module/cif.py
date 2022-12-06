# Ref: https://github.com/George0828Zhang/torch_cif
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from .cnn import SameConv2d
from .transformer.encoder_module import Linear
from .transformer.masking import len_to_mask


def pad2(x: Tensor, value: Tensor, size: Tuple[int, int]) -> Tensor:
    # x: (B, T, D)
    # value: (1, 1, D)
    value = value.expand(x.shape[0], -1, -1)
    pad_list = []
    if size[0] > 0:
        pad_list += [value] * size[0]
    pad_list.append(x)
    if size[1] > 0:
        pad_list += [value] * size[1]

    return torch.cat(pad_list, dim=1)  # (B, T+size[0]+size[1], D)


def prob_check(tensor, eps=1e-10, neg_inf=-1e8, logp=False):
    assert not torch.isnan(tensor).any(), "Nan in a probability tensor."
    # Add the eps here to prevent errors introduced by precision
    if logp:
        assert tensor.le(0).all() and tensor.ge(neg_inf).all(), (
            "Incorrect values in a log-probability tensor" ", -inf <= tensor <= 0"
        )
    else:
        assert tensor.le(1.0 + eps).all() and tensor.ge(0.0 - eps).all(), (
            "Incorrect values in a probability tensor" ", 0.0 <= tensor <= 1.0"
        )


def cif_function(
    input: Tensor,
    alpha: Tensor,
    beta: float = 1.0,
    tail_thres: float = 0.5,
    padding_mask: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    eps: float = 1e-4,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""A fast parallel implementation of continuous integrate-and-fire (CIF)
    https://arxiv.org/abs/1905.11235

    Args:
        input (Tensor): (N, S, C) Input features to be integrated.
        alpha (Tensor): (N, S) Weights corresponding to each elements in the
            input. It is expected to be after sigmoid function.
        beta (float): the threshold used for determine firing.
        tail_thres (float): the threshold for determine firing for tail handling.
        padding_mask (Tensor, optional): (N, S) A binary mask representing
            padded elements in the input.
        target_lengths (Tensor, optional): (N,) Desired length of the targets
            for each sample in the minibatch.
        eps (float, optional): Epsilon to prevent underflow for divisions.
            Default: 1e-4

    Returns -> Dict[str, List[Optional[Tensor]]]: Key/values described below.
        cif_out (Tensor): (N, T, C) The output integrated from the source.
        cif_lengths (Tensor): (N,) The output length for each element in batch.
        alpha_sum (Tensor): (N,) The sum of alpha for each element in batch.
            Can be used to compute the quantity loss.
        delays (Tensor): (N, T) The expected delay (in terms of source tokens) for
            each target tokens in the batch.
        tail_weights (Tensor, optional): (N,) During inference, return the tail.
    """
    B, S, C = input.size()
    assert tuple(alpha.size()) == (B, S), f"{alpha.size()} != {(B, S)}"
    # prob_check(alpha)

    dtype = alpha.dtype
    alpha = alpha.float()
    if padding_mask is not None:
        padding_mask = padding_mask.bool()
        alpha = alpha.masked_fill(padding_mask, 0)

    if target_lengths is not None:
        feat_lengths = target_lengths.long()
        desired_sum = beta * target_lengths.type_as(input) + eps
        alpha_sum = alpha.sum(1)
        alpha = alpha * (desired_sum / alpha_sum).unsqueeze(1)
        T = feat_lengths.max()
    else:
        alpha_sum = alpha.sum(1)
        feat_lengths = (alpha_sum / beta).floor().long()
        T = feat_lengths.max()

    # aggregate and integrate
    csum = alpha.cumsum(-1)
    with torch.no_grad():
        # indices used for scattering
        right_idx = (csum / beta).floor().long().clip(max=T)
        left_idx = right_idx.roll(1, dims=1)
        left_idx[:, 0] = 0

        # count # of fires from each source
        fire_num = right_idx - left_idx
        extra_weights = (fire_num - 1).clip(min=0)

    # The extra entry in last dim is for
    output = input.new_zeros((B, T + 1, C))
    delay = input.new_zeros((B, T + 1))
    source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(input)
    zero = alpha.new_zeros((1,))

    # right scatter
    fire_mask = fire_num > 0
    right_weight = torch.where(
        fire_mask, csum - right_idx.type_as(alpha) * beta, zero
    ).type_as(input)
    # assert right_weight.ge(0).all(), f"{right_weight} should be non-negative."
    output.scatter_add_(
        1, right_idx.unsqueeze(-1).expand(-1, -1, C), right_weight.unsqueeze(-1) * input
    )
    delay.scatter_add_(1, right_idx, right_weight * source_range / beta)

    # left scatter
    left_weight = (alpha - right_weight - extra_weights.type_as(alpha) * beta).type_as(
        input
    )
    output.scatter_add_(
        1, left_idx.unsqueeze(-1).expand(-1, -1, C), left_weight.unsqueeze(-1) * input
    )
    delay.scatter_add_(1, left_idx, left_weight * source_range / beta)

    # extra scatters
    if extra_weights.ge(0).any():
        extra_steps = extra_weights.max().item()
        tgt_idx = left_idx
        src_feats = input * beta
        for _ in range(extra_steps):
            tgt_idx = (tgt_idx + 1).clip(max=T)
            # (B, S, 1)
            src_mask = extra_weights > 0
            output.scatter_add_(
                1,
                tgt_idx.unsqueeze(-1).expand(-1, -1, C),
                src_feats * src_mask.unsqueeze(2),
            )
            delay.scatter_add_(1, tgt_idx, source_range * src_mask)
            extra_weights -= 1

    # tail handling
    if target_lengths is not None:
        # training time -> ignore tail
        output = output[:, :T, :]
        delay = delay[:, :T]
    else:
        # find out contribution to output tail
        # note: w/o scaling, extra weight is all 0
        zero = right_weight.new_zeros((1,))
        r_mask = right_idx == feat_lengths.unsqueeze(1)
        tail_weights = torch.where(r_mask, right_weight, zero).sum(-1)
        l_mask = left_idx == feat_lengths.unsqueeze(1)
        tail_weights += torch.where(l_mask, left_weight, zero).sum(-1)

        # a size (B,) mask that extends position that passed threshold.
        extend_mask = tail_weights >= tail_thres

        # extend 1 fire and upscale the weights
        if extend_mask.any():
            # (B, T, C), may have infs so need the mask
            upscale = (
                torch.ones_like(output)
                .scatter(
                    1,
                    feat_lengths.view(B, 1, 1).expand(-1, -1, C),
                    beta
                    / tail_weights.masked_fill(~extend_mask, beta)
                    .view(B, 1, 1)
                    .expand(-1, -1, C),
                )
                .detach()
            )
            output *= upscale
            feat_lengths += extend_mask.long()
            T = feat_lengths.max()
        output = output[:, :T, :]
        delay = delay[:, :T]

        # a size (B, T) mask to erase weights
        tail_mask = torch.arange(T, device=output.device).unsqueeze(
            0
        ) >= feat_lengths.unsqueeze(1)
        output[tail_mask] = 0

    fire_b, fire_t = fire_mask.nonzero(as_tuple=True)
    indices = [[] for _ in range(B)]
    for b, t in zip(fire_b, fire_t):
        indices[b].append(t - 1)
    return {
        "cif_out": output,
        "cif_lengths": feat_lengths,
        "alpha_sum": alpha_sum.to(dtype),
        "delays": delay,
        "tail_weights": tail_weights if target_lengths is None else [],
        "indices": indices,
    }


class CIF(nn.Module):
    def __init__(
        self, threshold: float = 1.0, normalize: bool = True, downsample: float = 0.0
    ):
        super().__init__()

        self.threshold = threshold  # beta in paper
        self.normalize = normalize
        self.downsample = downsample

    def forward(
        self,
        x: Tensor,
        x_len: LongTensor,
        prob: Tensor,
        tgt_len: LongTensor = None,
    ):
        # Check
        assert x.dim() == 3, x.shape
        assert x_len.dim() == 1, x_len.shape
        assert prob.dim() == 2, prob.shape
        if tgt_len is not None:
            assert tgt_len.dim() == 1, tgt_len.shape
            tgt_len = tgt_len.type(prob.dtype)

        # x:       (B, T, D)
        # x_len:   (B, )
        # prob:    (B, T)
        # tgt_len: (B, )

        B, T, _ = x.shape
        # device = x.device

        mask = len_to_mask(x_len)
        prob = prob * mask.type(prob.dtype)
        quantity_loss = 0.0

        if self.downsample > 0.0 and self.normalize:
            assert tgt_len is None
            with torch.no_grad():
                tgt_len = x_len.type(prob.dtype) / self.downsample
        if tgt_len is not None:
            # Compute quantity loss
            quantity_loss = (prob.sum(1) - tgt_len).abs().mean()
            # for b in range(B):
            #     quantity_loss += (prob[b, : x_len[b]].sum() - tgt_len[b]).abs()
            # quantity_loss /= B
        if tgt_len is not None and self.normalize:
            # Normalize weights
            for b in range(B):
                prob[b, : x_len[b]] *= tgt_len[b] / prob[b, : x_len[b]].sum()

        prob_1 = torch.cat(
            [torch.zeros((B, 1), dtype=prob.dtype, device=prob.device), prob], dim=1
        )  # (B, T+1)
        cum_prob = torch.cumsum(prob_1, dim=1)
        with torch.no_grad():
            # prob_detach = prob_1.detach()
            # cum_prob = torch.cumsum(prob_detach, dim=1)
            dis_prob = (
                torch.div(cum_prob.detach(), self.threshold, rounding_mode="floor")
                * self.threshold
            )
            dif_prob = dis_prob[:, 1:] - dis_prob[:, :-1]  # (B, T)
            fire_b, fire_t = torch.nonzero(dif_prob, as_tuple=True)
            # Note: fire_b & fire_t are indiced from 0

        prev_b, prev_t = 0, 1
        h_list = [[] for _ in range(B)]
        fired_indices = [[] for _ in range(B)]
        for b, t in zip(fire_b, fire_t):
            t_1 = t + 1
            if b != prev_b:
                prev_b = b
                prev_t = 1
            if t_1 > x_len[b]:
                prev_t = 1
                continue

            remained_weight = cum_prob[b, prev_t] - dis_prob[b, prev_t]
            h = remained_weight * x[b, prev_t - 1]

            if prev_t + 1 < t_1:
                h += (
                    prob_1[b, prev_t + 1 : t_1].unsqueeze(-1) * x[b, prev_t : t_1 - 1]
                ).sum(0)

            last_weight = dis_prob[b, t_1] - cum_prob[b, t_1 - 1]
            h += last_weight * x[b, t_1 - 1]

            h_list[b].append(h)
            fired_indices[b].append(t)
            prev_t = t_1

        out_len = []
        for b in range(B):
            h_list[b] = torch.stack(h_list[b], dim=0)
            out_len.append(len(h_list[b]))
        out = pad_sequence(h_list, batch_first=True)
        out_len = LongTensor(out_len).to(x_len.device)

        # prob_detach = prob.cpu().detach()
        # weights = prob.clone()
        # out_list, out_len = [], []
        # fired_indices = []
        # quantity_loss = 0.0
        # for b in range(B):
        #     h_list = []
        #     prev_t = 0
        #     accum = 0.0
        #     fired_indices.append([])
        #     for t in range(x_len[b]):
        #         accum += prob_detach[b, t].data
        #         if accum >= self.threshold:
        #             weights[b, t] = weights[b, t] + self.threshold - accum
        #             h = (x[b, prev_t : t + 1] * weights[b, prev_t : t + 1].unsqueeze(-1)).sum(0)
        #             h_list.append(h)
        #             weights[b, t] = (
        #                 accum - prob_detach[b, t].data + prob[b, t] - self.threshold
        #             )
        #             prev_t = t + 0
        #             fired_indices[b].append(t)
        #     h_b = torch.stack(h_list, dim=0)  # (T', D)
        #     out_list.append(h_b)
        #     out_len.append(len(h_b))
        #     if tgt_len is not None:
        #         quantity_loss += (prob[b, : x_len[b]].sum() - tgt_len[b]).abs()
        # out = pad_sequence(out_list, batch_first=True)
        # out_len = LongTensor(out_len).to(x_len.device)
        # quantity_loss /= B

        return out, out_len, quantity_loss, fired_indices


class DownsampleCIF(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        threshold: float = 1.0,
        normalize: bool = True,
        downsample: float = 4.0,
        dropout: float = 0.1,
        calc_weight: str = "original",
        window_size: int = 5,
        **kwargs,
    ):
        super().__init__()

        self.out_dim = hid_dim
        # self.feat_proj = Linear(in_dim, hid_dim)
        self.feat_proj = SameConv2d(in_dim, 32, hid_dim)

        self.calc_weight = calc_weight
        if calc_weight == "original":
            # self.conv = nn.Conv1d(hid_dim, hid_dim, 3, 1, 1)
            self.conv = nn.Conv1d(in_dim, hid_dim, 3, 1, 1)
            self.fc = Linear(hid_dim, 1)
        elif calc_weight.startswith("sim"):
            self.calc_weight, self.sim_metric = calc_weight.split("-")
            self.window_size = window_size
            self.half_window = window_size // 2
            assert window_size % 2 == 1, window_size
            assert self.sim_metric in {"l2", "cosine", "inner"}, self.sim_metric

            self.sim_proj = Linear(hid_dim, hid_dim)
            self.weight_func = nn.Sequential(
                Linear(self.window_size, hid_dim), nn.ReLU(), Linear(hid_dim, 1)
            )
            self.pad_emb = nn.Parameter(0.1 * torch.randn(1, 1, hid_dim))

        self.dropout = nn.Dropout(dropout)

        # self.cif = CIF(threshold, normalize, downsample=downsample)
        self.downsample = downsample

    def forward(
        self,
        x: Tensor,
        x_len: LongTensor,
        tgt_len: LongTensor = None,
    ):
        x = self.feat_proj(x)
        # (B, T, D)
        B, T, _ = x.shape

        if self.calc_weight == "original":
            out = self.conv(x.transpose(1, 2)).transpose(1, 2)
            out = torch.relu(out)
            out = self.dropout(out)
            out = self.fc(out).squeeze(-1)
            prob = torch.sigmoid(out)
        elif self.calc_weight == "sim":
            out = self.sim_proj(x)
            out = pad2(out, self.pad_emb, (self.half_window, self.half_window))
            if self.sim_metric == "l2":
                # sim = -(out.unsqueeze(1) - out.unsqueeze(2)).pow(2).sum(-1)
                sim_flat = [
                    (
                        out[:, self.half_window : self.half_window + T]
                        - out[:, i : i + T]
                    )
                    .pow(2)
                    .sum(-1)
                    for i in range(self.window_size)
                ]
            elif self.sim_metric == "cosine":
                sim = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(2), dim=-1)
            elif self.sim_metric == "inner":
                # sim = (out.unsqueeze(1) * out.unsqueeze(2)).sum(-1)
                sim_flat = [
                    (
                        out[:, self.half_window : self.half_window + T]
                        * out[:, i : i + T]
                    ).sum(-1)
                    for i in range(self.window_size)
                ]
                # sim_flat: [(B, T)] * window_size

            # sim: (B, T+W-1, T+W-1)
            # sim_flat = [
            #     sim.diagonal(i, 1, 2)[:, self.half_window + i :]
            #     for i in range(-self.half_window, self.half_window + 1)
            # ]

            sim_flat = torch.stack(sim_flat, dim=2)
            assert sim_flat.shape[1] == x.shape[1], sim_flat.shape
            assert sim_flat.shape[2] == self.window_size, sim_flat.shape

            out = self.weight_func(sim_flat).squeeze(-1)
            prob = torch.sigmoid(out)

        # x = self.feat_proj(x)
        # out, out_len, quantity_loss, fired_indices = self.cif(x, x_len, prob, tgt_len)

        tgt_len = torch.div(x_len, self.downsample, rounding_mode="floor").long()
        res = cif_function(
            x, prob, padding_mask=~len_to_mask(x_len), target_lengths=tgt_len
        )
        out = res["cif_out"]
        out_len = res["cif_lengths"]
        fired_indices = res["indices"]

        tgt_len = tgt_len.type(prob.dtype)
        quantity_loss = 0.0
        for b in range(B):
            quantity_loss += (prob[b, : x_len[b]].sum() - tgt_len[b]).abs()
        quantity_loss /= B

        return {
            "x": out,
            "x_len": out_len,
            "loss": quantity_loss,
            "prob": prob,
            "indices": fired_indices,
        }
