#!/bin/bash
cd /mnt/data/kw/hjy/EasyR1
set -x
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONUNBUFFERED=1

MODEL_PATH=/mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  # replace it with your local file path

screen -L -Logfile /mnt/data/kw/hjy/logs/0907/1.5B_linear.log python3 -m verl.trainer.main \
    config=/mnt/data/kw/hjy/EasyR1/examples/hjy_config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=/mnt/data/kw/hjy/EasyR1/examples/reward_function/math_linear.py:compute_score