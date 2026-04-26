#!/bin/bash
# run_train.sh

export CUDA_VISIBLE_DEVICES=2,3
export MASTER_PORT=29503

torchrun \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    /home/boran.lbr/gitspace/github/minimind/trainer/train_full_sft.py \
    --batch_size=32 \
    --epochs=3

# 执行：bash run_train.sh
