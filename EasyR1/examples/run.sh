#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTHONUNBUFFERED=1

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  # replace it with your local file path

python3 -m verl.trainer.main \
    config=config.yaml \
    trainer.max_steps=150 \
    data.shuffle=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=JET \
    trainer.save_freq=10 \
    trainer.val_freq=10 \
    trainer.n_gpus_per_node=4 \
    trainer.save_checkpoint_path=../ckp/JET_ckp \
    worker.reward.reward_function=./reward_function/math.py:compute_score