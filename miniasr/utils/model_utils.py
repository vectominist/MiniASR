"""
    File      [ model_utils.py ]
    Author    [ Heng-Jui Chang (MIT CSAIL) ]
    Synopsis  [ Some utilities for models. ]
"""

import torch
from easydict import EasyDict as edict


def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """Un-freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True


def check_extractor_attributes(model, device="cuda:0"):
    """Checks extractor's attributes."""

    print(f"Testing {model} on {device}")

    extractor = torch.hub.load("s3prl/s3prl", model).to(device)
    extractor.eval()
    wavs = [torch.zeros(160000, dtype=torch.float).to(device) for _ in range(8)]
    with torch.no_grad():
        hiddens = extractor(wavs)
        print(f"Hidden feature types: {hiddens.keys()}")

    from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)

    print(
        "Used {:.2f} MB of GPU VRAM with 8 audio samples each having 10 secs.".format(
            info.used / 1024 / 1024
        )
    )


def load_from_checkpoint(
    checkpoint, device="cpu", pl_ckpt=False, decode_args=None, mode=None
):
    """Loads model from checkpoint."""

    if isinstance(checkpoint, str):
        # Load from path
        ckpt = torch.load(checkpoint, map_location=device)
        args = edict(ckpt["args"])
        tokenizer = ckpt["tokenizer"]
        state_dict = ckpt["state_dict"]
    else:
        # Load from a loaded checkpoint
        args = edict(checkpoint["args"])
        tokenizer = checkpoint["tokenizer"]
        state_dict = checkpoint["state_dict"]

    if decode_args is not None:
        args.decode = decode_args
    if mode is not None:
        args.mode = mode

    if args.model.name == "ctc_asr":
        from miniasr.model.ctc_asr import ASR
    else:
        raise NotImplementedError(
            "{} ASR type is not supported.".format(args.model.name)
        )

    if not pl_ckpt:
        model = ASR(tokenizer, args).to(device)
        model.load_state_dict(state_dict, strict=False)
    else:
        assert isinstance(checkpoint, str)
        del ckpt, state_dict
        model = ASR.load_from_checkpoint(
            checkpoint_path=checkpoint, tokenizer=tokenizer, args=args
        ).to(device)

    return model, args, tokenizer


if __name__ == "__main__":
    import sys

    check_extractor_attributes(sys.argv[1])
