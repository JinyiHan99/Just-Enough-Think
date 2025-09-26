import subprocess

# Merge batch
for i in range(10, 21, 10):
    command = f"python EasyR1/scripts/model_merger.py \
        --local_dir ckp/7B/global_step_{i}/actor"
    print(f"Running: {command}")
    completed_process = subprocess.run(command, shell=True)
    if completed_process.returncode != 0:
        print(f"Script failed with params!!!")
    else:
        print(f"Script succeeded with params!!!")