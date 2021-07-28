'''
    File      [ eval_metrics.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Evaluation metrics. ]
'''

import edit_distance as ed


def split_sequence(seq, mode='word'):
    ''' Split sequence by word or characters. '''

    if mode in ['word', 'phone']:
        return seq.split(' ')
    elif mode == 'char':
        return list(seq)
    else:
        raise NotImplementedError(f'Unknown mode {mode}')


def sequence_distance(ref, hyp, mode='word'):
    ''' Calculates distance between two sequences '''

    ref, hyp = split_sequence(ref, mode), split_sequence(hyp, mode)
    sm = ed.SequenceMatcher(a=ref, b=hyp)

    return {
        'length': len(ref),
        'distance': sm.distance()
    }


def sequence_distance_full(ref, hyp, mode='word'):
    ''' Calculates distance between two sequences.
        (including substitution, deletion, and insertion) '''

    ref, hyp = split_sequence(ref, mode), split_sequence(hyp, mode)
    sm = ed.SequenceMatcher(a=ref, b=hyp)
    opcodes = sm.get_opcodes()

    Sub = sum([(x[2] - x[1] if x[0] == 'replace' else 0) for x in opcodes])
    Del = sum([(x[2] - x[1] if x[0] == 'delete' else 0)
               for x in opcodes])
    Ins = sum([(x[4] - x[3] if x[0] == 'insert' else 0)
               for x in opcodes])

    return {
        'length': len(ref),
        'distance': sm.distance(),
        'sub': Sub,
        'del': Del,
        'ins': Ins
    }


def print_eval_error_rates(
        n_sent, n_tok, sub_rate, del_rate, ins_rate, err_rate, sent_err):
    ''' Show results after testing. '''

    formatter = '| ' + ('{:<9}' * 2) + '| ' + ('{:<7}' * 5) + '|'
    prec = '{:.1f}'
    print(formatter.format(
        '#Snt', '#Tok', 'Sub', 'Del', 'Ins', 'Err', 'SErr'))
    print(formatter.format(
        n_sent, n_tok,
        prec.format(sub_rate * 100.),
        prec.format(del_rate * 100.),
        prec.format(ins_rate * 100.),
        prec.format(err_rate * 100.),
        prec.format(sent_err * 100.)
    ))
    print()
