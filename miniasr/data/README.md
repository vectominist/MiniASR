# Data

## `audio.py`

Audio and acoustic feature processing functions, including `load_waveform` to load raw waveforms from files. A simple implementation of [SpecAugment](https://arxiv.org/abs/1904.08779) is in `SpecAugment`.


## `text.py`

Tokenizers for processing text data, including reading vocabulary files, string-index conversion. Character, word, and subword units are supported.


## `dataset.py`

A standard `torch.utils.data.Dataset` for ASR is in `ASRDataset`. It simply loads preprocessed data list and tokenizes text data.


## `dataloader.py`

Creates `torch.utils.data.DataLoader` for different scenarios (train/dev/test).
