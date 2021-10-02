# LibriSpeech

The [LibriSpeech](https://www.openslr.org/12) corpus is composed of 1000 hours of spoken books. The corpus is partitioned into several parts: `train-clean-100`, `train-clean-360`, `train-other-500`, `dev-clean`, `dev-other`, `test-clean`, and `test-other`.

## Run

Modify paths in `path.sh`, then run all scripts as
```bash
bash preprocess.sh
bash train.sh
bash test.sh
```

## hubert_base + BLSTM + CTC + 4-gram LM (960h)

* Model
    * Extractor: hubert_base (95M params, fixed)
    * Encoder: 2-layer BLSTM 1024 units per direction (40M params)
* Data
    * Training: `train-clean-100` + `train-clean-360` + `train-other-500`
    * Vocab: character
* Decode
    * LM: official `4-gram.arpa.gz`
    * Beam size: 100
* Config
    * Train: `config/ctc_train_960h.yaml`
    * Test: `config/ctc_test_960h.yaml`
* Computation
    * FP: 16 bit
    * GPU: NVIDIA Tesla V100 32GB * 1
    * Time: 48h


**With LM**

```
| #Snt     #Tok     | Sub    Del    Ins    Err    SErr   |
dev-clean
| 2703     54402    | 2.2    0.2    0.3    2.8    33.2   |
test-clean
| 2620     52576    | 2.4    0.2    0.5    3.1    34.5   |
dev-other
| 2864     50948    | 6.4    0.6    0.9    7.8    53.2   |
test-other
| 2939     52343    | 6.2    0.6    0.9    7.7    57.5   |
```

**Without LM**

```
| #Snt     #Tok     | Sub    Del    Ins    Err    SErr   |
dev-clean
| 2703     54402    | 3.7    0.3    0.3    4.3    45.8   |
test-clean
| 2620     52576    | 3.8    0.3    0.4    4.5    46.6   |
dev-other
| 2864     50948    | 9.9    0.9    0.7    11.6   68.3   |
test-other
| 2939     52343    | 9.8    0.9    0.8    11.5   71.2   |
```


## hubert_base + BLSTM + CTC + 4-gram LM (100h)

* Model
    * Extractor: hubert_base (95M params, fixed)
    * Encoder: 2-layer BLSTM 1024 units per direction (40M params)
* Data
    * Training: `train-clean-100`
    * Vocab: character
* Decode
    * LM: official `4-gram.arpa.gz`
    * Beam size: 100
* Config
    * Train: `config/ctc_train_100h.yaml`
    * Test: `config/ctc_test_100h.yaml`
* Computation
    * FP: 16 bit
    * GPU: NVIDIA Tesla V100 32GB * 1
    * Time: 18h


**With LM**

```
| #Snt     #Tok     | Sub    Del    Ins    Err    SErr   |
dev-clean
| 2703     54402    | 3.0    0.4    0.3    3.7    42.2   |
test-clean
| 2620     52576    | 3.3    0.4    0.4    4.1    43.1   |
dev-other
| 2864     50948    | 7.9    1.1    0.7    9.7    61.5   |
test-other
| 2939     52343    | 8.0    1.1    0.8    9.9    65.4   |
```

**Without LM**

```
| #Snt     #Tok     | Sub    Del    Ins    Err    SErr   |
dev-clean
| 2703     54402    | 5.1    0.4    0.4    5.9    56.9   |
test-clean
| 2620     52576    | 5.4    0.4    0.5    6.4    58.0   |
dev-other
| 2864     50948    | 13.1   1.1    1.0    15.1   77.9   |
test-other
| 2939     52343    | 12.9   1.0    1.1    15.1   79.2   |
```
