"""
    File      [ timit.py ]
    Author    [ Heng-Jui Chang (MIT) ]
    Synopsis  [ Preprocess the TIMIT corpus. ]
"""

from pathlib import Path


def read_text(file: str):
    # Read transcriptions
    with open(file.replace(".wav", ".txt"), "r") as fp:
        text_wrd = fp.readline()
        text_wrd = " ".join(text.strip().split(" ")[2:])

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
            align_phn.append((t_1, t_2, phone))
            phn_list.append(phone)
    text_phn = " ".join(phn_list)

    return {
        "text": text_wrd,
        "phone": text_phn,
        "align_word": align_wrd,
        "align_phone": align_phn,
    }


def find_data(root: str):
    """
    Find all files in TIMIT.
    Output:
        data_dict [dict]:
            {
                'audio file idx': {
                    'file': audio file name
                    'text': transcription
                }
            }
    """

    # Find all audio files
    audio_list = list(Path(root).rglob("*.wav"))
    audio_list = sorted([str(f) for f in audio_list])

    # Find all transcriptions & merge to a data_dict
    data_dict = {}
    for file in audio_list:
        # Get file's idx
        # e.g. /data/sls/d/corpora/original/timit/train/dr1/fcjf0/sa1.wav
        #      -> train/dr1/fcjf0/sa1
        file_idx = "/".join(file.split("/")[-4:])
        data_dict[file_idx] = read_text(file)
        data_dict[file_idx]["file"] = file

    return data_dict
