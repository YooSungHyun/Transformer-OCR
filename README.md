# Transformer-OCR by Torch-Lightning

This Source code is converted https://github.com/him4318/Transformer-ocr to Torch-Lightning

so, core source is same that git repo!



Before use, chk your gpu count and change some scripts plz!!!!!!

I didn't final launch test after clean code, so some code error can be occurred



this source not need vocab.json, because `models/transformer_ocr/tokenizer.py` in it

but I write `make_vocab.py` when if you need🤗

## Data Usage

Plz see data/README.md

## Training Script Usage

1. cd your project root(./pytorch-lightning-template)

```
# Don't Script RUN in your scripts FOLDER!!!!! CHK PLZ!!!!!!!
bash scripts/run_train_~~~.sh
```

## Inference Script Usage

1. cd your project root(./pytorch-lightning-template)

```
# Don't Script RUN in your scripts FOLDER!!!!! CHK PLZ!!!!!!!
bash scripts/run_inference~~~.sh
```

# (Optional) Install DeepSpeed

1. run pip_install_deepspeed.sh

```
bash pip_install_deepspeed.sh
```