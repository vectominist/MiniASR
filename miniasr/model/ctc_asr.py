"""
    File      [ ctc_asr.py ]
    Author    [ Heng-Jui Chang (MIT CSAIL) ]
    Synopsis  [ CTC ASR model. ]
"""

import logging

import numpy as np
import torch
from easydict import EasyDict
from torch import nn

from miniasr.model.base_asr import BaseASR
from miniasr.module import (
    DownsampleCIF,
    DownsampleConv2d,
    RNNEncoder,
    TransformerEncoder,
)


class ASR(BaseASR):
    """
    CTC ASR model
    """

    def __init__(self, tokenizer, args: EasyDict):
        super().__init__(tokenizer, args)

        # Conv Layer
        hid_dim = self.in_dim
        self.cnn = None
        if self.args.model.get("cnn", None) is not None:
            self.cnn = DownsampleConv2d(self.in_dim, **args.model.cnn)
            hid_dim = self.cnn.out_dim
        elif self.args.model.get("cif", None) is not None:
            self.cif = DownsampleCIF(self.in_dim, **args.model.cif)
            hid_dim = self.cif.out_dim

        # Encoder Layer
        if self.args.model.encoder.module in {"RNN", "GRU", "LSTM"}:
            self.encoder = RNNEncoder(hid_dim, **args.model.encoder)
        elif self.args.model.encoder.module in {"transformer", "conformer"}:
            self.encoder = TransformerEncoder(hid_dim, **args.model.encoder)
        else:
            raise NotImplementedError(
                f"Unkown encoder module {self.args.model.encoder.module}"
            )

        self.ctc_output_layer = nn.Linear(self.encoder.out_dim, self.vocab_size)

        # Loss function (CTC loss)
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

        # Beam decoding with Flashlight
        self.enable_beam_decode = False
        if (
            self.args.mode in {"dev", "test"}
            and self.args.get("decode", {}).get("type", "greedy") == "beam"
        ):
            self.enable_beam_decode = True
            self.setup_flashlight()

    def setup_flashlight(self):
        """
        Setup flashlight for beam decoding.
        """
        import math

        from flashlight.lib.text.decoder import (
            CriterionType,
            KenLM,
            LexiconDecoder,
            LexiconDecoderOptions,
            SmearingMode,
            Trie,
        )
        from flashlight.lib.text.dictionary import (
            Dictionary,
            create_word_dict,
            load_words,
        )

        token_dict = Dictionary(self.args.decode.token)
        lexicon = load_words(self.args.decode.lexicon)
        word_dict = create_word_dict(lexicon)

        lm = KenLM(self.args.decode.lm, word_dict)

        sil_idx = token_dict.get_index("|")
        unk_idx = word_dict.get_index("<unk>")

        trie = Trie(token_dict.index_size(), sil_idx)
        start_state = lm.start(False)

        for word, spellings in lexicon.items():
            usr_idx = word_dict.get_index(word)
            _, score = lm.score(start_state, usr_idx)
            for spelling in spellings:
                # convert spelling string into vector of indices
                spelling_idxs = [token_dict.get_index(token) for token in spelling]
                trie.insert(spelling_idxs, usr_idx, score)
        trie.smear(SmearingMode.MAX)

        options = LexiconDecoderOptions(
            self.args.decode.beam_size,
            self.args.decode.token_beam_size,
            self.args.decode.beam_threshold,
            self.args.decode.lm_weight,
            self.args.decode.word_score,
            -math.inf,
            self.args.decode.sil_score,
            self.args.decode.log_add,
            CriterionType.CTC,
        )

        blank_idx = token_dict.get_index("#")  # for CTC
        is_token_lm = False  # we use word-level LM
        self.flashlight_decoder = LexiconDecoder(
            options, trie, lm, sil_idx, blank_idx, unk_idx, [], is_token_lm
        )
        self.token_dict = token_dict

        logging.info(
            f"Beam decoding with beam size {self.args.decode.beam_size}, "
            f"LM weight {self.args.decode.lm_weight}, "
            f"Word score {self.args.decode.word_score}"
        )

    def forward(self, wave, wave_len):
        """
        Forward function to compute logits.
        Input:
            wave [list]: list of waveform files
            wave_len [long tensor]: waveform lengths
        Output:
            logtis [float tensor]: Batch x Time x Vocabs
            enc_len [long tensor]: encoded length (logits' lengths)
            feat [float tensor]: extracted features
            feat_len [long tensor]: length of extracted features
        """

        other = {}

        # Extract features
        feat, feat_len = self.extract_features(wave, wave_len)

        # CNN/CIF features
        if self.cnn:
            feat, feat_len = self.cnn(feat, feat_len)
        elif self.cif:
            res = self.cif(feat, feat_len)
            feat, feat_len = res["x"], res["x_len"]
            other["quantity_loss"] = res["loss"]
            other["cif_prob"] = res["prob"]
            other["cif_indices"] = res["indices"]

        # Encode features
        if self.args.model.encoder.module in {"RNN", "GRU", "LSTM"}:
            enc, enc_len = self.encoder(feat, feat_len)
        if self.args.model.encoder.module in {"transformer", "conformer"}:
            enc, _other = self.encoder(feat, feat_len)
            enc_len = feat_len
            other = {**other, **_other}

        # Project hidden features to vocabularies
        logits = self.ctc_output_layer(enc)

        return logits, enc_len, feat, feat_len, other

    def cal_loss(self, logits, enc_len, feat, feat_len, text, text_len):
        """Computes CTC loss."""

        log_probs = torch.log_softmax(logits, dim=2)

        # Compute loss
        with torch.backends.cudnn.flags(deterministic=True):
            # for reproducibility
            ctc_loss = self.ctc_loss(log_probs.transpose(0, 1), text, enc_len, text_len)

        return ctc_loss

    def decode(self, logits, enc_len, decode_type=None):
        """Decoding."""
        if self.enable_beam_decode and decode_type != "greedy":
            return self.beam_decode(logits, enc_len)
        return self.greedy_decode(logits, enc_len)

    def greedy_decode(self, logits, enc_len):
        """CTC greedy decoding."""
        hyps = torch.argmax(logits, dim=2).cpu().tolist()  # Batch x Time
        return [
            self.tokenizer.decode(h[: enc_len[i]], ignore_repeat=True)
            for i, h in enumerate(hyps)
        ]

    def beam_decode(self, logits, enc_len):
        """Flashlight beam decoding."""

        greedy_hyps = self.greedy_decode(logits, enc_len)
        log_probs = torch.log_softmax(logits, dim=2) / np.log(10)

        beam_hyps = []
        for i, log_prob in enumerate(log_probs):
            emissions = log_prob.cpu()
            hyps = self.flashlight_decoder.decode(
                emissions.data_ptr(), enc_len[i], self.vocab_size
            )

            if len(hyps) > 0 and hyps[0].score < 10000.0:
                hyp = self.tokenizer.decode(hyps[0].tokens, ignore_repeat=True)
                beam_hyps.append(hyp.strip())
            else:
                beam_hyps.append(greedy_hyps[i])

        return beam_hyps
