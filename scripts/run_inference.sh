#!/bin/bash
GPU_IDS=3

model_path=model_outputs/ocr-accuracy=0.7395-epoch=21-val_loss=0.4672.ckpt

CUDA_VISIBLE_DEVICES=$GPU_IDS \
python3 inference.py \
    --seed=42 \
    --test_data_path=./preprocess/test.csv \
    --model_path=$model_path/checkpoint/model.bin \
    --config_path=config/model_config.json \
    --accelerator=gpu \
    --strategy=ddp \
    --devices=1