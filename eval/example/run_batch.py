import subprocess
task_name = [
    'math_500',
    'gpqa_diamond_instruct',
    'aime24_avg',
    'commonsenseqa',
    'gsm8k'
]

for i in task_name:
    for j in range(20, 41, 10):
        # command = f"CUDA_VISIBLE_DEVICES=7 screen -L -Logfile /mnt/data/kw/tom/log/0908.log lighteval vllm \
        #     'model_name=/mnt/data/kw/hjy/ckp/0906_easyr1/global_step_{j}/actor/huggingface,max_model_length=20000,generation_parameters={"max_new_tokens"=16000,"temperature"=0.6,"top_p"=0.95}' \
        #     --output-dir /mnt/data/kw/tom/main_experiments/0906_easyr1/step_{j} \
        #     --save-details \
        #     'community|{i}|0|0' \
        #     --custom-tasks /mnt/data/kw/tom/lighteval/community_tasks/task_lcr1.py"
        model_args = (
            f"'model_name=/mnt/data/kw/hjy/ckp/0907_1.5B_main/global_step_{j}/actor/huggingface,"
            f"max_model_length=20000,"
            f'generation_parameters={{"max_new_tokens":16000,"temperature":0.6,"top_p":0.95}}'
            "'"
        )
        command = (
            f"CUDA_VISIBLE_DEVICES=5 screen -L -Logfile /mnt/data/kw/tom/log/0908_task_{i}_step_{j}.log lighteval vllm \\\n"
            f"    {model_args} \\\n"
            f"    --output-dir /mnt/data/kw/tom/main_experiments/0907_1.5B_main/step_{j} \\\n"
            f"    --save-details \\\n"
            f"    'community|{i}|0|0' \\\n"
            f"    --custom-tasks /mnt/data/kw/tom/lighteval/community_tasks/task_lcr1.py"
        )
        print(f"Running: {command}")
        completed_process = subprocess.run(command, shell=True)
            # 检查执行情况 
        if completed_process.returncode != 0:
            print(f"Script failed with params!!!")
        else:
            print(f"Script succeeded with params!!!")



        