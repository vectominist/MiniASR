'''
    File      [ scheduler.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Learning rate scheduler. ]
'''

from torch.optim.lr_scheduler import LambdaLR


def create_lambda_lr_warmup(
        optimizer, warmup_step=5000, decay_type='fixed', max_step=-1):
    '''
        Creates a lambda learning rate scheduler.
    '''

    assert warmup_step > 0

    if decay_type == 'fixed':
        # No decay
        def lr_func(step):
            return min(step / warmup_step, 1.)
    elif decay_type == 'linear':
        # Linear decays to zero
        assert max_step > warmup_step

        def lr_func(step):
            if step <= warmup_step:
                return step / warmup_step
            return 1. - (step - warmup_step) / (max_step - warmup_step)
    elif decay_type == 'poly':
        # Decays with a power of -0.5
        def lr_func(step):
            if step <= warmup_step:
                return step / warmup_step
            return (warmup_step / step) ** 0.5
    else:
        raise NotImplementedError(f'Unknown decay type {decay_type}')

    return LambdaLR(optimizer, lr_func)
