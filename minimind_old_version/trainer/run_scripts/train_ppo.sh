#!/bin/bash
# run_train.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29503

torchrun \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    /home/boran.lbr/gitspace/github/minimind/trainer/train_ppo.py

