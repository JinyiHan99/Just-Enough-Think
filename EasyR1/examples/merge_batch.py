import subprocess

for i in range(110, 171, 10):
    command = f"python /mnt/data/kw/hjy/EasyR1/scripts/model_merger.py \
        --local_dir /mnt/data/kw/hjy/ckp/0907_1.5B_main/global_step_{i}/actor"
    print(f"Running: {command}")
    completed_process = subprocess.run(command, shell=True)
        # 检查执行情况
    if completed_process.returncode != 0:
        print(f"Script failed with params!!!")
    else:
        print(f"Script succeeded with params!!!")