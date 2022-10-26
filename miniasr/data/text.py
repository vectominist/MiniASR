"""
    File      [ text.py ]
    Author    [ Heng-Jui Chang (MIT CSAIL) ]
    Synopsis  [ Tokenizer for text data.
                Modified from tensorflow_datasets.features.text.*
                Ref: https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text_lib  ]
"""

import abc
import re


class _BaseTextEncoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, s):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, idxs, ignore_repeat=False):
        raise NotImplementedError

    @abc.abstractproperty
    def vocab_size(self):
        raise NotImplementedError

    @abc.abstractproperty
    def token_type(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def load_from_file(cls, vocab_file):
        raise NotImplementedError

    @property
    def pad_idx(self):
        return 0

    @property
    def eos_idx(self):
        return 1

    @property
    def unk_idx(self):
        return 2

    def __repr__(self):
        return "<{} vocab_size={}>".format(type(self).__name__, self.vocab_size)


class CharacterTextEncoder(_BaseTextEncoder):
    """Tokenizer for character-level tokens."""

    def __init__(self, vocab_list):
        # Note that vocab_list must not contain <pad>, <eos> and <unk>
        # <pad>=0, <eos>=1, <unk>=2
        self._vocab_list = ["<pad>", "<eos>", "<unk>"] + vocab_list
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}

    def encode(self, s):
        # Always strip trailing space, \r and \n
        s = s.strip("\r\n ")
        if self.vocab_to_idx(" ") == self.unk_idx:
            s = "".join(s.split(" "))
        # Manually append eos to the end
        return [self.vocab_to_idx(v) for v in s] + [self.eos_idx]

    def decode(self, idxs, ignore_repeat=False):
        vocabs = []
        for t, idx in enumerate(idxs):
            if idx == self.eos_idx:
                break
            if idx == self.pad_idx or (
                ignore_repeat and t > 0 and idx == idxs[t - 1 if t > 0 else 0]
            ):
                continue
            v = self.idx_to_vocab(idx)
            vocabs.append(v)
        out = "".join(vocabs)
        return re.sub(" +", " ", out)

    @classmethod
    def load_from_file(cls, vocab_file):
        with open(vocab_file, "r") as fp:
            # Do not strip space because character based text encoder should
            # have a space token
            vocab_list = [line.strip("\r\n") for line in fp]
        return cls(vocab_list)

    @property
    def vocab_size(self):
        return len(self._vocab_list)

    @property
    def token_type(self):
        return "character"

    def vocab_to_idx(self, vocab):
        return self._vocab2idx.get(vocab, self.unk_idx)

    def idx_to_vocab(self, idx):
        return self._vocab_list[idx]


class SubwordTextEncoder(_BaseTextEncoder):
    """Tokenizer for subword-level tokens."""

    def __init__(self, spm):
        if spm.pad_id() != 0 or spm.eos_id() != 1 or spm.unk_id() != 2:
            raise ValueError(
                "Please train sentencepiece model with following argument:\n"
                "--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 --model_type=bpe --eos_piece=<eos>"
            )
        self.spm = spm
        self._vocab_list = [spm.id_to_piece(i) for i in range(spm.get_piece_size())]

    def encode(self, s):
        return self.spm.encode_as_ids(s)

    def decode(self, idxs, ignore_repeat=False):
        crop_idx = []
        for t, idx in enumerate(idxs):
            if idx == self.eos_idx:
                break
            elif idx == self.pad_idx or (
                ignore_repeat and t > 0 and idx == idxs[t - 1]
            ):
                continue
            else:
                crop_idx.append(int(idx))

        return self.spm.decode_ids(crop_idx)

    @classmethod
    def load_from_file(cls, filepath):
        import sentencepiece as splib

        spm = splib.SentencePieceProcessor()
        spm.load(filepath)
        spm.set_encode_extra_options(":eos")
        return cls(spm)

    @property
    def vocab_size(self):
        return len(self.spm)

    @property
    def token_type(self):
        return "subword"


class WordTextEncoder(CharacterTextEncoder):
    """Tokenizer for word-level tokens."""

    def encode(self, s):
        # Always strip trailing space, \r and \n
        s = s.strip("\r\n ")
        # Space as the delimiter between words
        words = s.split(" ")
        # Manually append eos to the end
        return [self.vocab_to_idx(v) for v in words if v != ""] + [self.eos_idx]

    def decode(self, idxs, ignore_repeat=False):
        vocabs = []
        for t, idx in enumerate(idxs):
            v = self.idx_to_vocab(idx)
            if idx == self.eos_idx:
                break
            if idx == self.pad_idx or (ignore_repeat and t > 0 and idx == idxs[t - 1]):
                continue
            vocabs.append(v)
        return " ".join(vocabs)

    @property
    def token_type(self):
        return "word"


class PhoneTextEncoder(WordTextEncoder):
    """Tokenizer for phoneme-level tokens."""

    @property
    def token_type(self):
        return "phone"


def load_text_encoder(mode, vocab_file):
    """Creates a text tokenizer."""
    if mode == "character":
        return CharacterTextEncoder.load_from_file(vocab_file)
    elif mode == "subword":
        return SubwordTextEncoder.load_from_file(vocab_file)
    elif mode in "word":
        return WordTextEncoder.load_from_file(vocab_file)
    elif mode in "phone":
        return PhoneTextEncoder.load_from_file(vocab_file)
    else:
        raise NotImplementedError(mode)
