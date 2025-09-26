import subprocess
task_name = [
    'math_500',
    'gpqa_diamond_instruct',
    'aime24_avg',
    'commonsenseqa',
    'gsm8k',
    'AMC_avg',
    'Olympiad',
    'mmlu_pro'
]

for i in task_name:
    for j in range(20, 41, 10):
        model_args = (
            f"'model_name=/mnt/data/kw/hjy/ckp/0907_1.5B_main/global_step_{j}/actor/huggingface,"   
            f"max_model_length=20000,"
            f'generation_parameters={{"max_new_tokens":16000,"temperature":0.6,"top_p":0.95}}'
            "'"
        )
        command = (
            f"CUDA_VISIBLE_DEVICES=5 screen -L -Logfile ./log/res.log lighteval vllm \\\n"
            f"    {model_args} \\\n"
            f"    --output-dir ./our_results \\\n"
            f"    --save-details \\\n"
            f"    'community|{i}|0|0' \\\n"
            f"    --custom-tasks ./eval/lighteval/community_tasks/task_lcr1.py"
        )
        print(f"Running: {command}")
        completed_process = subprocess.run(command, shell=True)
        if completed_process.returncode != 0:
            print(f"Script failed with params!!!")
        else:
            print(f"Script succeeded with params!!!")



        