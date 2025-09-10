
import re
import json

from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

def correct_fn(answer, ground_truth):
    ground_truth_str = str(ground_truth)
    pattern = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'
    boxed_content_ans = re.findall(pattern, answer)

    if not boxed_content_ans:
        return 0
    final_ans_expr = "\\boxed{" + boxed_content_ans[-1] + "}"
    if "\\boxed" not in ground_truth_str:
        final_gt_expr = "\\boxed{" + ground_truth_str + "}"
    else:
        final_gt_expr = ground_truth_str
    if final_ans_expr == final_gt_expr:
        return 1.0
    try:
        parsed_ans = parse(final_ans_expr)
        parsed_gt = parse(final_gt_expr)
        is_correct = verify(parsed_ans, parsed_gt)
        return 1.0 if is_correct else 0
    except Exception as e:
        return 0


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



from transformers import AutoTokenizer

def calculate_token_length(texts, tokenizer):
 
    token_lengths = []
    for t in texts:
        tokens = tokenizer.encode(t, add_special_tokens=False)
        token_lengths.append(len(tokens))
    return token_lengths




    

def cal_metrics(data_path, ans_key, std_key, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = read_jsonl(data_path)
    # print(len())
    hit = 0
    total = len(data)

    total_length = 0
    for example in data:
        ans = example[ans_key]['text'][0]
        std = example[std_key]
        ## cal the acc
        eval_res = correct_fn(ans, std)
        if eval_res > 0:
            hit += 1
        ## cal the token length
        ans_length = len(tokenizer.encode(ans, add_special_tokens=False))
        # import pdb
        # pdb.set_trace()
        total_length += ans_length
    
    acc = hit / total
    average_token_length = total_length / total
    print(f"{data_path}\n {total} {acc * 100:.2f}%  token_length: {average_token_length:.2f} ")
    return acc, average_token_length

data_path = "/mnt/data/kw/hjy/ckp/0907_1.5B_main/global_step_30/actor/huggingface/8aafb857b068ef5c/community|math_500|0_results_with_std.jsonl"
cal_metrics(data_path,"sample","std",model_path= "/mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    


