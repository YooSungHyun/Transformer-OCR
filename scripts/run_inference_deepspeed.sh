#!/bin/bash
GPU_IDS="0,1,2,3"
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export WANDB_DISABLED=false
export TOKENIZERS_PARALLELISM=false


model_path=model_outputs/ocr-accuracy=0.7395-epoch=21-val_loss=0.4672.ckpt
python3 $model_path/zero_to_fp32.py $model_path $model_path/checkpoint/model.bin

if [ $? -eq "0" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_IDS \
        python3 -m torch.distributed.launch --nproc_per_node=4 inference_deepspeed.py \
                --seed=42 \
                --test_data_path=./preprocess/test.csv \
                --model_path=$model_path/checkpoint/model.bin \
                --config_path=config/model_config.json \
                --accelerator=gpu \
                --strategy=ddp \
                --devices=4 \
                --num_nodes=1
fi

if [ $? -eq "0" ]; then
        python3 ddp_inference_gather.py
fi