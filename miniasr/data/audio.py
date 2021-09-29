'''
    File      [ audio.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Audio processing. ]
'''

import numpy as np
import torch
import torchaudio
from librosa import resample


def load_waveform(file, target_sample_rate=16_000):
    '''
        Loads raw waveform from file
        Input:
            file [str]: path to audio file
            target_sample_rate [int]: specified sample rate
        Output:
            waveform [torch.Tensor]: raw waveform tensor
    '''

    try:
        waveform, sample_rate = torchaudio.load(
            file, normalize=True, channels_first=True)
    except:
        waveform, sample_rate = torchaudio.load(
            file, normalization=True, channels_first=True)
    # waveform: Channel x Time
    waveform = waveform.mean(0)  # Averaged by channels

    if sample_rate != target_sample_rate:
        waveform = torch.FloatTensor(
            resample(waveform.numpy(), sample_rate,
                     target_sample_rate, res_type='kaiser_best'))

    return waveform


class SpecAugment():
    '''
        SpecAugment
        https://arxiv.org/abs/1904.08779
        https://arxiv.org/abs/1912.05533
    '''

    def __init__(
            self,
            freq_mask_range=[0, 30], freq_mask_num=2,
            time_mask_range=[0, 100], time_mask_num=2, time_mask_max=1.0,
            time_warp_w=80):

        self.freq_mask_range = freq_mask_range
        self.freq_mask_num = freq_mask_num
        self.time_mask_range = time_mask_range
        self.time_mask_num = time_mask_num
        self.time_mask_max = time_mask_max
        self.time_warp_w = time_warp_w

    def time_warping(self, feat):
        tau = feat.shape[1]
        w = np.random.randint(-self.time_warp_w, self.time_warp_w + 1)
        w0 = np.random.randint(self.time_warp_w, tau - self.time_warp_w)
        mapping = np.arange(tau, dtype=float)
        mapping[:w0 + 1] = mapping[:w0 + 1] * (w0 + w) / w0
        mapping[w0 + 1:] = (mapping[w0 + 1:] * (tau - 1 - w0 - w) +
                            (tau - 1) * w) / (tau - 1 - w0)
        mapping = np.clip(np.around(mapping).astype(int), 0, tau)
        return feat[:, mapping, :]

    def frequency_masking(self, feat):
        f = np.random.randint(
            self.freq_mask_range[0], self.freq_mask_range[1])
        f0 = np.random.randint(0, feat.shape[2] - f)
        feat[:, :, f0:f0 + f] = 0

    def time_masking(self, feat):
        max_t = int(feat.shape[1] * self.time_mask_max)
        t = np.random.randint(
            self.time_mask_range[0],
            min(self.time_mask_range[1], max_t))
        t0 = np.random.randint(0, feat.shape[1] - t)
        feat[:, t0:t0 + t, :] = 0

    def __call__(self, feat):
        '''
            Applies SpecAugment to feat
            feat [float tensor]: Batch x Time x Freq
        '''

        if self.time_warp_w > 0:
            feat = self.time_warping(feat)

        if self.freq_mask_num > 0:
            for _ in range(self.freq_mask_num):
                self.frequency_masking(feat)

        if self.time_mask_num > 0:
            for _ in range(self.time_mask_num):
                self.time_masking(feat)

        return feat
