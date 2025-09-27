import subprocess
task_name = [
    # 'gpqa_diamond_instruct',
    # 'Olympiad',
    # 'commonsenseqa',
    # 'aime24_avg',
    # 'mmlu',
    # 'gsm8k',
    # 'AMC_avg',
    'math500'
]

for i in task_name:
    model_args = (
        f"'model_name=/mnt/data/kw/hjy/ckp/0923_7B_hw/global_step_130/actor/huggingface,"   
        f"max_model_length=20000,tensor_parallel_size=1,"
        f'generation_parameters={{"max_new_tokens":16000,"temperature":0.6,"top_p":0.95}}'
        "'"
    )
    command = (
        f"CUDA_VISIBLE_DEVICES=0 lighteval vllm \\\n"
        f"    {model_args} \\\n"
        f"    --output-dir ./our_results \\\n"
        f"    --save-details \\\n"
        f"    'community|{i}|0|0' \\\n"
        f"    --custom-tasks ./eval/lighteval/community_tasks/task_JET.py"
    )
    print(f"Running: {command}")
    completed_process = subprocess.run(command, shell=True)
    if completed_process.returncode != 0:
        print(f"Script failed with params!!!")
    else:
        print(f"Script succeeded with params!!!")