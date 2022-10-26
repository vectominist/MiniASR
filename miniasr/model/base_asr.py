"""
    File      [ base_asr.py ]
    Author    [ Heng-Jui Chang (MIT CSAIL) ]
    Synopsis  [ Base ASR model. ]
"""

import logging
import time
from os.path import join

import pytorch_lightning as pl
import torch
from easydict import EasyDict
from s3prl.nn import Featurizer, S3PRLUpstream

from miniasr.data.audio import SpecAugment
from miniasr.module import create_lambda_lr_warmup
from miniasr.utils import (
    freeze_model,
    print_eval_error_rates,
    sequence_distance,
    sequence_distance_full,
)


def get_model_stride(name: str):
    """Returns the stride of a model (in ms)"""
    return 20 if any(m in name.split("_") for m in ["hubert", "wav2vec2"]) else 10


def extracted_length(length: int, stride: int = 10):
    """Calculates extracted feature's length."""
    if stride == 20:
        return (length // 16 - 30) // 20
    return (length // 16 - 15) // 10


class BaseASR(pl.LightningModule):
    """
    Base ASR model (no actual functionality)
        tokenizer [_BaseTextEncoder]: text tokenizer
        args [EasyDict]: arguments
    """

    def __init__(self, tokenizer, args: EasyDict):
        super().__init__()

        self.save_hyperparameters(dict(args=args))

        # Param setting
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        # Load feature extractor (from s3prl)
        extractor = S3PRLUpstream(args.model.extractor.name)
        self.extractor_stride = get_model_stride(args.model.extractor.name)
        self.enable_train_extractor = args.model.extractor.train
        self.extractor = extractor
        if not self.enable_train_extractor:
            # Freeze the extractor if not enabled for fine-tuning
            freeze_model(self.extractor)
        if self.extractor_stride == 20:
            # Disable layerdrop for hubert and wav2vec2
            self.extractor.model.encoder.layerdrop = 0

        # Feature selection module
        self.feat_select = Featurizer(
            extractor,
            args.model.extractor.get("layer_selections", None),
            args.model.extractor.get("normalize", False),
        )
        self.in_dim = self.feat_select.output_size

        # Data augmentation
        self.specaug = None
        if args.model.get("specaugment", None):
            self.specaug = SpecAugment(**args.model.specaugment)

        # Timer setup (for testing)
        self.time_count = 0.0
        self.audio_duration = 0.0

    def configure_optimizers(self):
        """Sets optimizer."""
        optimizer = getattr(torch.optim, self.args.model.optim.algo)(
            self.parameters(), **self.args.model.optim.kwargs
        )
        if self.args.model.optim.get("scheduler", None):
            scheduler = create_lambda_lr_warmup(
                optimizer, **self.args.model.optim.scheduler
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": 1,
                    "interval": "step",
                },
            }
        else:
            return optimizer

    def extract_features(self, wave, wave_len):
        """
        Extract acoustic features from raw waveforms
        Input:
            wave [list]: waveforms
            wave_len [long tensor]: waveform lengths
        Output:
            feat [float tensor]: feature vectors (Batch x Time x Dim)
            feat_len [long tensor]: lengths of features (Batch)
        """

        # Extract features
        if self.enable_train_extractor:
            feat, feat_len = self.extractor(wave, wave_len)
        else:
            with torch.no_grad():
                feat, feat_len = self.extractor(wave, wave_len)

        # Get features
        feat, feat_len = self.feat_select(feat, feat_len)

        # Data augmentation
        if self.training and self.specaug is not None:
            feat = self.specaug(feat)

        return feat, feat_len

    def forward(self, wave, wave_len):
        """Forward function to compute logits."""
        raise NotImplementedError
        # Should return logits, enc_len, feat, feat_len

    def cal_loss(self, logits, enc_len, feat, feat_len, text, text_len):
        """Computes loss."""
        raise NotImplementedError
        # Should return loss

    def training_step(self, batch, batch_idx):
        """Processes in a single training loop."""

        # Get inputs from batch
        wave, text = batch["wave"], batch["text"]
        wave_len, text_len = batch["wave_len"], batch["text_len"]

        # Compute logits
        logits, enc_len, feat, feat_len, other = self(wave, wave_len)

        # Compute loss
        loss = self.cal_loss(logits, enc_len, feat, feat_len, text, text_len)
        self.log("train_ctc_loss", loss)

        if "kmeans_loss" in other:
            # VQ
            kmeans_loss = torch.stack(other["kmeans_loss"], dim=0).sum()
            loss += kmeans_loss
            self.log("train_kmeans_loss", kmeans_loss)
            if "code_perplexity" in other:
                for i, ppl in enumerate(other["code_perplexity"]):
                    self.log(f"code_ppl/layer_{i}", ppl)
        if "quantity_loss" in other:
            # CIF
            quantity_loss = other["quantity_loss"]
            loss += quantity_loss
            self.log("train_quantity_loss", quantity_loss)

        # Log information
        self.log("train_loss", loss)

        return loss

    def decode(self, logits, enc_len, decode_type=None):
        """Decodes output logits."""
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """
        Processes in a single validation loop.
        Performs CTC decoding.
        """

        # Get inputs from batch
        wave, text = batch["wave"], batch["text"]
        wave_len, text_len = batch["wave_len"], batch["text_len"]

        with torch.no_grad():
            # Compute logits
            logits, enc_len, feat, feat_len, other = self(wave, wave_len)

            # Compute loss
            loss = self.cal_loss(logits, enc_len, feat, feat_len, text, text_len)

            # Decode (hypotheses)
            hyps = self.decode(logits, enc_len)

            # Recover reference text
            refs = [
                self.tokenizer.decode(text[i].cpu().tolist()) for i in range(len(text))
            ]

        return list(zip(refs, hyps)), loss.cpu().item()

    def validation_epoch_end(self, outputs):
        """
        End of validation.
        Computes CER, WER, and loss.
        """

        char_res = {"length": 0, "distance": 0}
        word_res = {"length": 0, "distance": 0}

        total_loss = 0  # Total loss
        total_samples = 0  # Total number of samples in val data

        for i, (out, loss) in enumerate(outputs):
            total_loss += loss * len(out)
            total_samples += len(out)
            for j, (ref, hyp) in enumerate(out):
                # Compute character error rate (CER)
                res = sequence_distance(ref, hyp, "char")
                for key, val in res.items():
                    char_res[key] += val

                # Compute character error rate (WER)
                res = sequence_distance(ref, hyp, "word")
                for key, val in res.items():
                    word_res[key] += val

                # Show some samples
                if (
                    i == len(outputs) // 2
                    and j < 5
                    and isinstance(
                        self.logger, pl.loggers.tensorboard.TensorBoardLogger
                    )
                ):
                    self.logger.experiment.add_text(
                        f"val_sample_{j}_ref", ref, self.global_step
                    )
                    self.logger.experiment.add_text(
                        f"val_sample_{j}_hyp", hyp, self.global_step
                    )

        # Compute CER and WER and save to log
        val_cer = char_res["distance"] / char_res["length"]
        val_wer = word_res["distance"] / word_res["length"]
        self.log("val_cer", val_cer)
        self.log("val_wer", val_wer)

        # Save averaged loss to
        val_loss = total_loss / total_samples
        self.log("val_loss", val_loss)

        print()
        logging.info(
            "Val CER = {:.1f}% , WER = {:.1f}% , Loss = {:.2f}".format(
                val_cer * 100, val_wer * 100, val_loss
            )
        )

    def test_step(self, batch, batch_idx):
        """Testing step."""

        time_begin = time.time()  # unit: seconds
        results = self.validation_step(batch, batch_idx)
        self.time_count += time.time() - time_begin
        self.audio_duration += batch["wave_len"].sum().cpu().item() / 16000

        return results

    def test_epoch_end(self, outputs):
        """
        End of testing.
        Computes CER, WER, and loss.
        """

        sent_err = 0  # Number of sentences with errors

        char_res = {"length": 0, "distance": 0, "sub": 0, "del": 0, "ins": 0}
        word_res = {"length": 0, "distance": 0, "sub": 0, "del": 0, "ins": 0}

        total_samples = 0  # Total number of samples in test data
        all_refs, all_hyps = [], []

        for out, _ in outputs:
            total_samples += len(out)
            for ref, hyp in out:
                # Collect results
                all_refs.append(ref)
                all_hyps.append(hyp)

                # Compute character error rate (CER)
                res = sequence_distance_full(ref, hyp, "char")
                for key, val in res.items():
                    char_res[key] += val

                # Compute character error rate (WER)
                res = sequence_distance_full(ref, hyp, "word")
                for key, val in res.items():
                    word_res[key] += val

                if res["distance"] > 0:
                    sent_err += 1

        # Compute CER and WER and save to log
        sent_err /= total_samples

        test_cer = char_res["distance"] / char_res["length"]
        test_c_sub = char_res["sub"] / char_res["length"]
        test_c_del = char_res["del"] / char_res["length"]
        test_c_ins = char_res["ins"] / char_res["length"]

        test_wer = word_res["distance"] / word_res["length"]
        test_w_sub = word_res["sub"] / word_res["length"]
        test_w_del = word_res["del"] / word_res["length"]
        test_w_ins = word_res["ins"] / word_res["length"]

        # Print results
        print("\n\nCharacter errors")
        print_eval_error_rates(
            total_samples,
            char_res["length"],
            test_c_sub,
            test_c_del,
            test_c_ins,
            test_cer,
            sent_err,
        )
        print("Word errors")
        print_eval_error_rates(
            total_samples,
            word_res["length"],
            test_w_sub,
            test_w_del,
            test_w_ins,
            test_wer,
            sent_err,
        )

        # Compute RTF and latency
        rtf = self.time_count / self.audio_duration
        latency = self.time_count / total_samples * 1000
        print("RTF:     {:.4f}".format(rtf))
        print("Latency: {:.4f} [ms/sentence]\n".format(latency))
        self.time_count = 0

        # Write results to file
        with open(join(self.args.test_res, "refs.txt"), "w") as fp:
            fp.write("\n".join(all_refs))
        with open(join(self.args.test_res, "hyps.txt"), "w") as fp:
            fp.write("\n".join(all_hyps))

    def recognize(self, wave):
        """
        Greedy decoding given a list of waveforms.
        Input:
            wave [list]: waveforms
        Output:
            hyps [list]: list of transcriptions
        """

        with torch.no_grad():
            wave_len = torch.LongTensor([len(w) for w in wave]).to(wave[0].device)
            logits, enc_len, _, _ = self(wave, wave_len)
            hyps = self.decode(logits, enc_len, "greedy")
            return hyps

    def on_save_checkpoint(self, checkpoint):
        """
        Additional information to be saved to checkpoint.
        """

        checkpoint["args"] = self.args
        checkpoint["tokenizer"] = self.tokenizer
