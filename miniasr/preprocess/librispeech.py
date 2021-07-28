'''
    File      [ librispeech.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Preprocess the LibriSpeech corpus. ]
'''

from pathlib import Path


def read_text(file):
    ''' Read transcriptions. (LibriSpeech) '''

    src_file = '-'.join(file.split('-')[:-1]) + '.trans.txt'

    with open(src_file, 'r') as fp:
        trans_dict = {}
        for line in fp:
            line = line.strip()
            idx = line.strip().split(' ')[0]
            trans_dict[idx] = {'text': line.split(' ', 1)[1]}
        return trans_dict


def find_data(root):
    '''
        Find all files in LibriSpeech.
        Output:
            data_dict [dict]:
                {
                    'audio file idx': {
                        'file': audio file name
                        'text': transcription
                    }
                }
    '''

    # Find all audio files
    audio_list = list(Path(root).rglob("*.flac"))
    audio_list = sorted([str(f) for f in audio_list])

    # Find all transcriptions & merge to a data_dict
    data_dict = {}
    for file in audio_list:
        # Get file's idx
        # e.g. /work/dataset/LibriSpeech/dev-clean/174/50561/174-50561-0011.flac
        #      -> 174-50561-0011
        file_idx = file.split('/')[-1].split('.')[0]
        if file_idx not in data_dict:
            trans_dict = read_text(file)
            for idx, val in trans_dict.items():
                data_dict[idx] = val
        data_dict[file_idx]['file'] = file

    return data_dict
