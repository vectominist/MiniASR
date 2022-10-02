"""
    File      [ timit.py ]
    Author    [ Heng-Jui Chang (MIT) ]
    Synopsis  [ Preprocess the TIMIT corpus. ]
"""

import os
from pathlib import Path

from .normalize_text import (
    remove_redundant_whitespaces,
    remove_strings,
    replace_strings,
)
from .timit_dev_spkr import dev_speakers
from .timit_phone import phone_map


def read_text(file: str, phone_size: int = 48):
    assert phone_size in {
        39,
        48,
        60,
    }, f"Phoneme size must be either 39, 48, or 60, not {phone_size}"

    # Read transcriptions
    with open(file.replace(".wav", ".txt"), "r") as fp:
        text_wrd = fp.readline()
        text_wrd = " ".join(text_wrd.strip().split(" ")[2:])
        text_wrd = text_wrd.upper()
        text_wrd = remove_strings(text_wrd, [".", "?", ",", ":", '"', "!", ";"])
        text_wrd = replace_strings(text_wrd, ["-"], [" "])
        text_wrd = remove_redundant_whitespaces(text_wrd)

    # Read alignment (word)
    with open(file.replace(".wav", ".wrd"), "r") as fp:
        align_wrd = []
        for line in fp:
            line = line.strip()
            if len(line) == 0:
                continue

            t_1 = line.split(" ")[0]
            t_2 = line.split(" ")[1]
            word = line.split(" ")[2]
            align_wrd.append((t_1, t_2, word))

    # Read alignment (phoneme)
    with open(file.replace(".wav", ".phn"), "r") as fp:
        phn_list = []
        align_phn = []
        for line in fp:
            line = line.strip()
            if len(line) == 0:
                continue

            t_1 = line.split(" ")[0]
            t_2 = line.split(" ")[1]
            phone = line.split(" ")[2]
            if phone_size != 60:
                phone = phone_map[phone][phone_size]
            if phone != "":
                align_phn.append((t_1, t_2, phone))
                phn_list.append(phone)
    text_phn = " ".join(phn_list)

    return {
        "text": text_wrd,
        "phone": text_phn,
        "align_word": align_wrd,
        "align_phone": align_phn,
    }


def find_data(root: str, split: str = "train"):
    """
    Find all files in TIMIT.
    Output:
        data_dict [dict]:
            {
                "audio file idx": {
                    "file": audio file path
                    "text": word transcription
                    "phone": phoneme transcription
                    "align_word": List[Tuple[int, int, str]] Audio-word alignment
                    "align_phone": List[Tuple[int, int, str]] Audio-phoneme alignment
                    "spkr": speaker id
                    "split": split name of the corpus
                }
            }
    """

    if split == "dev":
        root = os.path.join(root, "test")
    else:
        root = os.path.join(root, split)

    # Find all audio files
    audio_list = list(Path(root).rglob("*.wav"))
    audio_list = sorted([str(f) for f in audio_list])

    # Find all transcriptions & add to a data_dict
    data_dict = {}
    for file in audio_list:
        # Get file's idx
        # e.g. /data/sls/d/corpora/original/timit/train/dr1/fcjf0/sa1.wav
        #      -> train/dr1/fcjf0/sa1
        file_idx = "/".join(file.split("/")[-4:])[:-4]
        speaker_id = file_idx.split("/")[2]

        if file_idx in data_dict:
            continue

        if (
            (split == "test" and speaker_id not in dev_speakers)
            or (split == "dev" and speaker_id in dev_speakers)
            or (split == "train")
        ):
            data_dict[file_idx] = read_text(file)
            data_dict[file_idx]["file"] = file
            data_dict[file_idx]["spkr"] = speaker_id
            data_dict[file_idx]["split"] = split

    return data_dict
