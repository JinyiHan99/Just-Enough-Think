CUDA_VISIBLE_DEVICES=3 lighteval vllm \
    "model_name=/mnt/data/kw/hjy/ckp/0907_1.5B_main/global_step_90/actor/huggingface,max_model_length=20000,generation_parameters={\"max_new_tokens\":16000,\"temperature\":0.6,\"top_p\":0.95}" \
    --output-dir ./our_results/ \
    --save-details \
    'community|Olympiad|0|0' \
    --custom-tasks ./eval/lighteval/community_tasks/task_lcr1.py 