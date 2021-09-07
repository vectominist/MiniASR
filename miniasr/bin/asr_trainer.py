'''
    File      [ asr_trainer.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Creates ASR trainer. ]
'''

import logging
import pytorch_lightning as pl

from miniasr.data.dataloader import create_dataloader


def create_asr_trainer(args, device):
    '''
        Creates ASR model and trainer. (for training)
    '''

    # Load data & tokenizer
    tr_loader, dv_loader, tokenizer = create_dataloader(args)

    # Create ASR model
    logging.info(f'Creating ASR model (type = {args.model.name}).')
    if args.model.name == 'ctc_asr':
        from miniasr.model.ctc_asr import ASR
    else:
        raise NotImplementedError(
            '{} ASR type is not supported yet.'.format(args.model.name))

    model = ASR(tokenizer, args).to(device)

    # Create checkpoint callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.trainer.default_root_dir,
        **args.checkpoint_callbacks
    )

    # Create pytorch-lightning trainer
    trainer = pl.Trainer(
        accumulate_grad_batches=args.hparam.accum_grad,
        gradient_clip_val=args.hparam.grad_clip,
        callbacks=[checkpoint_callback],
        **args.trainer
    )

    return tr_loader, dv_loader, tokenizer, model, trainer


def create_asr_trainer_test(args, device):
    '''
        Creates ASR model and trainer. (for testing)
    '''

    # Load data & tokenizer
    _, dv_loader, tokenizer = create_dataloader(args)

    # Create ASR model
    logging.info(f'Creating ASR model (type = {args.model.name}).')
    if args.model.name == 'ctc_asr':
        from miniasr.model.ctc_asr import ASR
    else:
        raise NotImplementedError(
            '{} ASR type is not supported yet.'.format(args.model.name))

    # Load model from checkpoint
    model = ASR.load_from_checkpoint(
        args.ckpt, tokenizer=tokenizer, args=args).to(device)

    # Create pytorch-lightning trainer
    trainer = pl.Trainer(**args.trainer)

    return None, dv_loader, tokenizer, model, trainer
