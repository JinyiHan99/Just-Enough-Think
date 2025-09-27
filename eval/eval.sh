#!/bin/bash

tasks=(
    # 'gpqa_diamond_instruct'
    # 'Olympiad'
    # 'commonsenseqa'
    'aime24_avg'
    # 'mmlu'
    # 'gsm8k'
    # 'AMC_avg'
    # 'math500'
)



output_dir="./eval_results"

for task in "${tasks[@]}"; do
  echo "========== Running task: $task =========="

  CUDA_VISIBLE_DEVICES=0 lighteval vllm \
    "model_name=JET-7B,max_model_length=20000,tensor_parallel_size=1,generation_parameters={\"max_new_tokens\":16000,\"temperature\":0.6,\"top_p\":0.95}" \
    --output-dir "$output_dir" \
    --save-details \
    "community|${task}|0|0" \
    --custom-tasks ./eval/lighteval/community_tasks/task_JET.py

done
