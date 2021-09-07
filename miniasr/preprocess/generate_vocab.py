'''
    File      [ generate_vocab.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Generates vocabularies. ]
'''

from collections import Counter
import sentencepiece as splib


def generate_word_char_vocab(text_list, out_file, vocab_size=5000, mode='word'):
    ''' Generate word/character-level vocabularies. '''

    counter = Counter()
    if mode == 'word':
        for text in text_list:
            counter.update(text.split(' '))
    elif mode == 'char':
        for text in text_list:
            counter.update(text)
    else:
        raise NotImplementedError(f'Unknown mode {mode}.')

    # Select top 'vocab_size' words as vocabularies.
    vocab_list = sorted(
        counter.keys(), key=lambda k: counter[k], reverse=True)[:vocab_size]
    vocab_list = sorted(vocab_list)

    print(f'Found {len(counter)} {mode}s.')
    print(f'Selected {len(vocab_list)} vocabularies.')
    print(f'Saving {mode} vocabularies to {out_file}')

    with open(out_file, 'w') as fp:
        fp.write('\n'.join(vocab_list))


def generate_subword_vocab(
        text_file, out_file, model_type='unigram', vocab_size=5000,
        character_coverage=1.0):
    ''' Generate subword-level vocabularies with sentencepiece. '''

    print('Using sentencepiece to generate vocab file.')
    print(f'Model type = {model_type}')
    print(f'Vocab size = {vocab_size}')

    cmd = (
        f'--input={text_file} --model_prefix={out_file} '
        f'--model_type={model_type} '
        f'--vocab_size={vocab_size} '
        f'--character_coverage={character_coverage} '
        f'--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 '
        f'--eos_piece=<eos> --remove_extra_whitespaces=true '
    )

    splib.SentencePieceTrainer.Train(cmd)


def merge_word_subword_vocabs(word_vocab_file, subword_vocab_file, out_file):
    '''
        Merge word/char and subword vocab files.
        E.g. Chinese characters + English subwords
    '''

    try:
        import sentencepiece_model_pb2 as model
    except ImportError:
        logging.error(
            'Cannot find sentencepiece_model_pb2, '
            'perhaps take a look at '
            'https://github.com/google/sentencepiece/blob/master/python/add_new_vocab.ipynb')

    with open(word_vocab_file, 'r') as fp:
        word_tokens = fp.read().split('\n')

    m = model.ModelProto()
    with open(subword_vocab_file, 'rb') as fp:
        m.ParseFromString(fp.read())

    for token in word_tokens:
        if token == ' ':
            continue
        new_token = model.ModelProto().SentencePiece()
        new_token.piece = token
        new_token.score = 0
        m.pieces.append(new_token)

    with open(out_file, 'wb') as fp:
        fp.write(m.SerializeToString())

    spm = splib.SentencePieceProcessor()
    spm.load(out_file)

    print(f'Final vocab size = {len(spm)}')
