#!/bin/bash
GPU_IDS="0,1,2,3"
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export WANDB_DISABLED=false
export TOKENIZERS_PARALLELISM=false
export LOCAL_RANK=0

CUDA_VISIBLE_DEVICES=$GPU_IDS \
python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 ./train.py \
    --output_dir="model_outputs/" \
    --train_data_path="preprocess/train.csv" \
    --valid_data_path="preprocess/valid.csv" \
    --config_path="config/dense_model.json" \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --accumulate_grad_batches=1 \
    --max_epochs=10 \
    --log_every_n_steps=1 \
    --accelerator=gpu \
    --strategy=deepspeed_stage_2 \
    --num_nodes=1 \
    --replace_sampler_ddp=false \
    --devices=4 \
    --auto_scale_batch_size=false \
    --valid_on_cpu=false \
    --deepspeed_config=ds_config/zero2.json
