# Preprocess

To train an ASR model on corpora, some basic information should be extracted to the specified format. For each partition of a dataset, some files are required for training or evaluation:
* `data_list_sorted.json`
  A `.json` file that contains a list of paired audio files and transcriptions sorted by audio duration decreasingly.
* Vocabulary files
  Character and word-level vocabularies should be stored in a `.txt` file, while subword should be extracted by [sentencepiece](https://github.com/google/sentencepiece).

Some corpora require processing beforehand such as segmentation of audio files or normalization of transcriptions. You may refer to [kaldi/egs](https://github.com/kaldi-asr/kaldi/tree/master/egs) and [espnet/egs](https://github.com/espnet/espnet/tree/master/egs) for more information.

## Collect Corpus Data

Please refer to the example in `librispeech.py`. For each partition of the dataset, a dictionary `data_dict` should be constructed containing all audio files and transcriptions by a function `find_data`.

The `data_dict` should be in the format of:
```json
{
    "index of audio file 1": {
        "file": "path to audio file 1",
        "text": "transcription of audio file 1"
    },
    "index of audio file 2": {
        "file": "path to audio file 2",
        "text": "transcription of audio file 2"
    },
    ...
}
```

The `find_data` function is the function you have to implement for your corpus. The `find_data` function will be used by `run_preprocess.py`, so you have to add an `elif` option in line 68 of `run_preprocess.py`, i.e.,
```python
elif args.corpus == 'My Corpus':
    from miniasr.preprocess.MyCorpus import find_data
```

## Vocabulary

Extraction of vocabularies is implemented in `generate_vocab.py`. It supports character, word, and subword level tokens for ASR training. For a detailed usage of subword extraction, you may refer to [sentencepiece](https://github.com/google/sentencepiece).

Moreover, you can merge character/word vocabulary files with a subword vocabulary by the `merge_word_subword_vocabs` function. This might be useful when the dataset is code-switched, e.g., English subword + Chinese character.

