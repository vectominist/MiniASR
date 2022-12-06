"""
    File      [ dataloader.py ]
    Author    [ Heng-Jui Chang (MIT CSAIL) ]
    Synopsis  [ General dataset ]
"""

import json
import logging

from torch.utils.data import Dataset
from tqdm import tqdm

from .text import _BaseTextEncoder


class ASRDataset(Dataset):
    """
    General dataset for ASR
    paths [list]: Paths to preprocessed data dict (.json)
    tokenizer [_BaseTextEncoder]: text tokenizer (see data/tokenizer.py)
    """

    def __init__(self, paths, tokenizer: _BaseTextEncoder, mode="train", max_len=1600):
        super().__init__()

        # Load preprocessed dictionaries
        logging.info(f"Loading data from {paths}")
        data_list = []
        for path in paths:
            with open(path, "r") as fp:
                d_list = json.load(fp)
            data_list += d_list

        self.vocab_type = tokenizer.token_type
        logging.info(f"Vocab type: {self.vocab_type}")

        if self.vocab_type in {"char", "subword", "word"}:
            trans_key = "text"
        else:
            trans_key = "phone"

        self.mode = (
            mode
            if (
                (data_list[0].get(trans_key, None) is not None)
                and (tokenizer is not None)
            )
            else "wild"
        )

        if self.mode != "wild":
            # Tokenize text data
            # Note: 'wild' mode does not have transcription
            for data in tqdm(data_list):
                data["text"] = tokenizer.encode(data[trans_key])

        self.data_list = [d for d in data_list if len(d.get(trans_key, [0])) > 0]

        for key in ["align_phone", "align_word"]:
            if key in self.data_list[0]:
                for i, d in enumerate(self.data_list):
                    self.data_list[i][key] = [
                        int(float(t) / 160) for (_, t, _) in d[key][:-1]
                    ]

        logging.info(
            f"{len(self.data_list)} audio files found " f"(mode = {self.mode})"
        )

    def __getitem__(self, index):
        """Returns a single sample."""

        out_dict = {
            "file": self.data_list[index]["file"],
            "text": self.data_list[index]["text"],
        }

        if "align_word" in self.data_list[index]:
            out_dict["align_word"] = self.data_list[index]["align_word"]
        if "align_phone" in self.data_list[index]:
            out_dict["align_phone"] = self.data_list[index]["align_phone"]

        return out_dict

    def __len__(self):
        """Size of the dataset."""
        return len(self.data_list)
