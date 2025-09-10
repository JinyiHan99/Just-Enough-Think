import json

import pandas as pd


import pdb



def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_list_to_jsonl(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_list_to_json(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

for i in range(150, 171, 10):
    # 这里是结果文件
    res_path = f"/mnt/data/kw/tom/main_experiments/0907_1.5B_main/step_{i}/community|math_500|0_results.jsonl"
    data = read_jsonl(res_path)
    sorted_data = sorted(data, key=lambda x: int(x["sample_id"]))
    print(len(sorted_data))

    # 这里是我们的测试数据文件
    math_res_path = "/mnt/data/kw/tom/data_test/math/math500_test_cleaned.jsonl"
    test_data = read_jsonl(math_res_path)

    wrong = 0
    res = []
    for i in range(len(sorted_data)):
        example = sorted_data[i]
        if test_data[i]['question'] in example['sample']['input']:
            example['std'] = test_data[i]['std']
        else:
            wrong += 1
        res.append(example)
    ## hjy：wrong输出的结果一定要为0
    print(wrong)
    # pdb.set_trace()
    # 保存的文件就带了std字段
    save_list_to_jsonl(res, res_path.replace(".jsonl","_with_std.jsonl"))






