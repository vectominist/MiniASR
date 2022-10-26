#!/usr/bin/env python3
"""
    File      [ get_sample.py ]
    Author    [ Heng-Jui Chang (MIT CSAIL) ]
    Synopsis  [ Get a sample output from a pre-trained ASR. ]
"""

import argparse
import logging
import os
import pickle

import torch
import torchaudio
from easydict import EasyDict

from miniasr.bin.asr_trainer import create_asr_test
from miniasr.data.dataloader import audio_collate_fn, get_dev_dataset
from miniasr.utils import base_args, logging_args, set_random_seed


def parse_arguments() -> EasyDict:
    """Parses arguments from command line."""
    parser = argparse.ArgumentParser(
        "Get all results from a pre-trained ASR model given a sample."
    )

    parser.add_argument("data", type=str, help="Dataset directory.")
    parser.add_argument("ckpt", type=str, help="Checkpoint for testing.")
    parser.add_argument("out", type=str, help="Output directory.")
    parser.add_argument("--sample_id", "-i", type=int, default=0)

    parser = base_args(parser)  # Add basic arguments
    args = parser.parse_args()
    logging_args(args)  # Set logging

    args = EasyDict(vars(args))
    args.test = True
    args.mode = "dev"
    args.decode = EasyDict({"type": "greedy"})

    return args


def forward_once(args, model, device, dataset):
    logging.info(f"Starting forward ASR model for sample {args.sample_id}")
    raw_data = dataset[args.sample_id]
    data = audio_collate_fn([raw_data], "dev")
    wave, text = data["wave"].to(device), data["text"].to(device)
    wave_len, text_len = data["wave_len"].to(device), data["text_len"].to(device)

    output = {}
    output["file"] = data["file"][0]
    output["wave"] = wave
    logging.info(
        f"Forwarding with audio waveform {output['file']} of length {wave.shape[1]}"
    )

    if "align_word" in raw_data:
        logging.info("Found `align_word` in this sample")
        output["align_word"] = raw_data["align_word"]
    if "align_phone" in raw_data:
        logging.info("Found `align_word` in this sample")
        output["align_phone"] = raw_data["align_phone"]

    with torch.no_grad():
        # Extract features
        logging.info("Extracting features with S3PRL")
        feat, feat_len = model.extract_features(wave, wave_len)
        output["input_feat"] = feat
        output["input_feat_len"] = feat_len
        logging.info(f"Extracted feature shape: {feat.shape}")

        # CNN features
        if model.cnn:
            logging.info("Downsampling features with CNN")
            feat, feat_len = model.cnn(feat, feat_len)
            output["cnn_feat"] = feat
            output["cnn_feat_len"] = feat_len
        elif model.cif:
            logging.info("Downsampling features with CIF")
            res = model.cif(feat, feat_len)
            feat, feat_len = res["x"], res["x_len"]
            output["quantity_loss"] = res["loss"]
            output["cif_prob"] = res["prob"]
            output["cif_indices"] = res["indices"][0]

        # Encode features
        enc_type = model.args.model.encoder.module
        logging.info(f"Forwarding {enc_type} encoder")
        if enc_type in {"RNN", "GRU", "LSTM"}:
            enc, enc_len = model.encoder(feat, feat_len)
        if enc_type in {"transformer", "conformer"}:
            enc, _other = model.encoder(
                feat, feat_len, get_hidden=True, get_attention=True
            )
            enc_len = feat_len
            output = {**output, **_other}

        # Project hidden features to vocabularies
        logging.info("Computing output logits")
        logits = model.ctc_output_layer(enc)
        output["output_logits"] = logits

        # Compute loss
        logging.info("Computing loss")
        loss = model.cal_loss(logits, enc_len, feat, feat_len, text, text_len)
        output["loss"] = loss

        # Decode
        logging.info("Performing greedy decoding")
        hyp = model.decode(logits, enc_len)[0]
        ref = model.tokenizer.decode(text[0].cpu().tolist())
        output["hyp"] = hyp
        output["ref"] = ref
        logging.info(f"Hyp: {hyp}")
        logging.info(f"Ref: {ref}")

        # move tensors to cpu and convert to numpy
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                output[k] = v.cpu().numpy()
            if isinstance(v, list):
                output[k] = [
                    (x if not isinstance(x, torch.Tensor) else x.cpu().numpy())
                    for x in v
                ]
            print(k, type(output[k]))

    path = os.path.join(args.out, f"out_{args.sample_id}.pkl")
    logging.info(f"Saving results to {path}")
    with open(path, "wb") as fp:
        pickle.dump(output, fp)

    logging.info(f"Finished!")


def main():
    # Basic setup
    torch.multiprocessing.set_sharing_strategy("file_system")
    torchaudio.set_audio_backend("sox_io")

    # Parse arguments
    args = parse_arguments()
    assert args.ckpt != "none"
    os.makedirs(args.out, exist_ok=True)

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Get device
    device = torch.device("cpu" if args.cpu else "cuda:0")
    args, tokenizer, model = create_asr_test(args, device)
    dataset = get_dev_dataset([args.data], tokenizer)
    model.eval()
    forward_once(args, model, device, dataset)


if __name__ == "__main__":
    main()
