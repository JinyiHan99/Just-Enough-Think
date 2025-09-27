CUDA_VISIBLE_DEVICES=0 screen -L -Logfile ./log/res.log lighteval vllm \
    "model_name=/mnt/data/kw/hjy/ckp/0923_7B_hw/global_step_50/actor/huggingface,max_model_length=20000,tensor_parallel_size=1,generation_parameters={\"max_new_tokens\":16000,\"temperature\":0.6,\"top_p\":0.95}" \
    --output-dir /mnt/data/kw/tom/main_experiments/0923_7B_hw/step_50_aime \
    --save-details \
    'community|math500|0|0' \
    --custom-tasks ./eval/lighteval/community_tasks/task_JET.py