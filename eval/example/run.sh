CUDA_VISIBLE_DEVICES=4 screen -L -Logfile /mnt/data/kw/tom/log/0908.log lighteval vllm \
    --output-dir /mnt/data/kw/tom/main_experiments/0906_easyr1/step_70 \
    /mnt/data/kw/tom/main_experiments/run.yaml \
    --save-details \
    'community|math_500|0|0' \
    --custom-tasks /mnt/data/kw/tom/lighteval/community_tasks/task_lcr1.py

CUDA_VISIBLE_DEVICES=5 screen -L -Logfile /mnt/data/kw/tom/log/0908.log lighteval vllm \
    --output-dir /mnt/data/kw/tom/main_experiments/0906_easyr1/step_70 \
    /mnt/data/kw/tom/main_experiments/run.yaml \
    --save-details \
    'community|gpqa_diamond_instruct|0|0' \
    --custom-tasks /mnt/data/kw/tom/lighteval/community_tasks/task_lcr1.py

CUDA_VISIBLE_DEVICES=1 screen -L -Logfile /mnt/data/kw/tom/log/0908.log lighteval vllm \
    --output-dir /mnt/data/kw/tom/main_experiments/0906_easyr1/step_70 \
    /mnt/data/kw/tom/main_experiments/run.yaml \
    --save-details \
    'community|aime24_avg|0|0' \
    --custom-tasks /mnt/data/kw/tom/lighteval/community_tasks/task_lcr1.py

CUDA_VISIBLE_DEVICES=3 lighteval vllm \
    --output-dir /mnt/data/kw/tom/main_experiments/DS-Distill-1.5B_results \
    /mnt/data/kw/tom/main_experiments/run.yaml \
    --save-details \
    'community|commonsenseqa|0|0' \
    --custom-tasks /mnt/data/kw/tom/lighteval/community_tasks/task_lcr1.py

CUDA_VISIBLE_DEVICES=4 screen -L -Logfile /mnt/data/kw/tom/log/0908.log lighteval vllm \
    --output-dir /mnt/data/kw/tom/main_experiments/0906_easyr1/step_70 \
    /mnt/data/kw/tom/main_experiments/run.yaml \
    --save-details \
    'community|gsm8k|0|0' \
    --custom-tasks /mnt/data/kw/tom/lighteval/community_tasks/task_lcr1.py



CUDA_VISIBLE_DEVICES=4 lighteval vllm \
    "model_name=/mnt/data/kw/hjy/ckp/0906_easyr1/global_step_30/actor/huggingface,max_model_length=20000,generation_parameters={'max_new_tokens'=16000,'temperature'=0.6,'top_p'=0.95}" \
    --output-dir /mnt/data/kw/tom/main_experiments/0906_easyr1/step_30 \
    --save-details \
    'community|commonsenseqa|0|0' \
    --custom-tasks /mnt/data/kw/tom/lighteval/community_tasks/task_lcr1.py 

python /mnt/data/kw/tom/main_experiments/experiment_analyzer.py \
    --base_dir /mnt/data/kw/tom/main_experiments/0907_1.5B_main \
    --output_csv /mnt/data/kw/tom/main_experiments/0907_1.5B_main/summary.csv \
    --tokenizer_path /mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

python /mnt/data/kw/tom/main_experiments/math_500_process.py \
    --base_dir /mnt/data/kw/hjy/ckp/0906_easyr1 \
    --output_csv /mnt/data/kw/tom/main_experiments/0906_easyr1/math500.csv \
    --tokenizer_path /mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

python /mnt/data/kw/tom/main_experiments/eval_commonsenseqa.py \
    --base_dir /mnt/data/kw/tom/main_experiments/0906_easyr1/step_20/details/mnt/data/kw/hjy/ckp/0906_easyr1/global_step_20/actor/huggingface/2025-09-08T06-13-42.741874/details_community|commonsenseqa|0_2025-09-08T06-13-42.741874_results.jsonl \
    --output_csv /mnt/data/kw/tom/main_experiments/0906_easyr1/commonsenseqa.csv \
    --tokenizer_path /mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

CUDA_VISIBLE_DEVICES=2 lighteval vllm \
    "model_name=/mnt/data/kw/hjy/ckp/0907_1.5B_main/global_step_90/actor/huggingface,max_model_length=20000,generation_parameters={\"max_new_tokens\":16000,\"temperature\":0.6,\"top_p\":0.95}" \
    --output-dir /mnt/data/kw/tom/main_experiments/0906_easyr1/step_60 \
    --save-details \
    'community|AMC_avg|0|0' \
    --custom-tasks /mnt/data/kw/tom/lighteval/community_tasks/task_lcr1.py 

CUDA_VISIBLE_DEVICES=3 lighteval vllm \
    "model_name=/mnt/data/kw/hjy/ckp/0907_1.5B_main/global_step_90/actor/huggingface,max_model_length=20000,generation_parameters={\"max_new_tokens\":16000,\"temperature\":0.6,\"top_p\":0.95}" \
    --output-dir /mnt/data/kw/tom/main_experiments/0906_easyr1/step_60 \
    --save-details \
    'community|Olympiad|0|0' \
    --custom-tasks /mnt/data/kw/tom/lighteval/community_tasks/task_lcr1.py 