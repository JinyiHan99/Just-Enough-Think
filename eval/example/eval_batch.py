import json
import re
import numpy as np
import os
import sys
from collections import defaultdict

# ==============================================================================
# 核心评估逻辑 (基于我们最终完善的版本)
# ==============================================================================

def extract_predicted_answer(model_response_text):
    """
    从模型的文本输出中提取预测答案。
    核心逻辑：查找所有符合特定模式的答案，并选择位置最靠后的那一个，因为它代表模型的最终结论。
    """
    if not isinstance(model_response_text, str):
        return None
        
    # 每个元组是: (优先级, 正则表达式)
    # 优先级 0 是最高的。
    patterns = [
        (0, re.compile(r"\\boxed\{([A-Z])\}")),
        (1, re.compile(r"\bANSWER:\s*\*?\s*([A-Z])\b", re.IGNORECASE)),
        (2, re.compile(r"\bOption ([A-Z])\b", re.IGNORECASE)),
        (3, re.compile(r"\b(?:the\s+)?(?:correct|final|best)?\s*(?:answer|choice)\s*(?:is|would be|should be)\s*:?\s*([A-Z])\b", re.IGNORECASE)),
        (4, re.compile(r"</think>\s*([A-Z])\.\s+[a-zA-Z]{2,}")),
        (4, re.compile(r"^\s*([A-Z])\.\s+[a-zA-Z]{2,}", re.MULTILINE)),
    ]

    found_matches = [] # 将存储元组 (index, priority, letter)

    for priority, pattern in patterns:
        for match in pattern.finditer(model_response_text):
            try:
                letter = match.group(1).strip().upper()
                if letter and 'A' <= letter <= 'Z':
                    found_matches.append((match.start(), priority, letter[0]))
            except IndexError:
                continue
    
    if not found_matches:
        return None

    # 核心逻辑：主要按位置（index）排序，选择最后出现的。
    found_matches.sort(key=lambda x: (x[0], -x[1]))
    
    return found_matches[-1][2]

def evaluate_single_file_for_acc(file_path):
    """
    评估单个 JSONL 文件并仅返回准确率。
    """
    if not os.path.exists(file_path):
        return "File Not Found"
        
    total_count, correct_count = 0, 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                total_count += 1

                gold_index = data.get('doc', {}).get('gold_index')
                choices = data.get('doc', {}).get('choices')
                
                if gold_index is None or choices is None or not (0 <= gold_index < len(choices)):
                    continue
                    
                gold_answer_letter = choices[gold_index]
                model_response_text = "".join(data.get('model_response', {}).get('text', []))
                predicted_answer_letter = extract_predicted_answer(model_response_text)
                
                if predicted_answer_letter is not None and predicted_answer_letter == gold_answer_letter:
                    correct_count += 1
            except Exception:
                # 在批量处理时忽略单个文件的解析错误
                continue
    
    if total_count == 0:
        return "No Samples"
        
    accuracy = (correct_count / total_count) * 100
    return f"{accuracy:.2f}%"


# ==============================================================================
# 批处理主程序
# ==============================================================================

def batch_process(base_directory):
    """
    遍历基础目录，查找并评估所有checkpoint的gpqa和commonsenseqa文件。
    """
    # 使用 defaultdict 方便地创建嵌套字典
    results = defaultdict(lambda: {'gpqa': 'Not Found', 'commonsenseqa': 'Not Found'})

    print(f"开始扫描目录: {base_directory}\n")

    # os.walk 会递归地遍历所有子目录
    for dirpath, _, filenames in os.walk(base_directory):
        # 检查当前目录是否是 checkpoint 目录
        match = re.search(r'(step_\d+)', dirpath)
        if not match:
            continue
        
        checkpoint_name = match.group(1)

        for filename in filenames:
            # 只处理我们关心的结果文件
            if not filename.endswith('_results.jsonl'):
                continue

            dataset_name = None
            if 'gpqa' in filename:
                dataset_name = 'gpqa'
            elif 'commonsenseqa' in filename:
                dataset_name = 'commonsenseqa'
            
            # 如果是目标数据集，并且该checkpoint还未被评估过
            if dataset_name and results[checkpoint_name][dataset_name] == 'Not Found':
                full_path = os.path.join(dirpath, filename)
                print(f"  正在评估: {checkpoint_name} -> {dataset_name}...")
                accuracy = evaluate_single_file_for_acc(full_path)
                results[checkpoint_name][dataset_name] = accuracy

    return results

if __name__ == "__main__":
    # --- 配置 ---
    # 检查是否从命令行提供了目录路径，否则使用默认路径
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # 请在这里设置您的默认顶层实验目录
        base_dir = "/mnt/data/kw/tom/main_experiments/0907_1.5B_main"

    if not os.path.isdir(base_dir):
        print(f"错误: 目录不存在 -> {base_dir}")
        sys.exit(1)

    # --- 执行 ---
    final_results = batch_process(base_dir)

    # --- 打印总结报告 ---
    print("\n" + "="*50)
    print(" " * 15 + "评估总结报告")
    print("="*50)

    if not final_results:
        print("未找到任何 'step_XX' 目录或相关的结果文件。")
    else:
        # 按 checkpoint 数字大小排序
        sorted_checkpoints = sorted(final_results.keys(), key=lambda x: int(x.split('_')[1]))
        
        for checkpoint in sorted_checkpoints:
            data = final_results[checkpoint]
            print(f"\nCheckpoint: {checkpoint}")
            print(f"  - commonsenseqa ACC: {data['commonsenseqa']}")
            print(f"  - gpqa          ACC: {data['gpqa']}")
    
    print("="*50)