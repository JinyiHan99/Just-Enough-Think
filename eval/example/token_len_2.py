# 文件名: token_len_logger_final_v4.py (最终版 - 修正数据结构假设和正则)

import json
import argparse
import os
import re
import csv
import time
from tqdm import tqdm
from typing import List, Dict, Optional, DefaultDict
from collections import defaultdict
from transformers import AutoTokenizer

# =================== 日志记录辅助函数 (修正正则) ===================
def parse_info(path: str, tokenizer_path: str, base_model_prefix: str, is_base_model_flag: bool) -> Optional[Dict[str, str]]:
    """
    从文件和路径中解析所有元数据。
    - 修正了正则表达式以匹配 'details_...' 和时间戳。
    """
    filename = os.path.basename(path)
    
    # 核心修正：更新正则表达式以匹配您的文件名格式
    # details_  <suite>  | <task> :<subset>| <shots> _<timestamp>_results.jsonl
    match_lighteval = re.search(r"details_([^|]+)\|([^|:]+):?([^|]*)\|(\d+)_.*?_results\.jsonl$", filename)
    
    if not match_lighteval:
        print(f"  [解析警告] 文件名 '{filename}' 不符合预期的 'details_suite|task|..._results.jsonl' 模式。")
        return None
        
    suite, task_main, task_sub, _ = match_lighteval.groups()
    dataset_name = f"{suite}-{task_main}"
    if task_sub:
        dataset_name += f"-{task_sub}"
            
    clean_path = tokenizer_path.strip('/')
    
    if is_base_model_flag:
        step = "origin"
        model_part = os.path.basename(clean_path)
    else:
        head, tail = os.path.split(clean_path)
        step = "origin" 
        model_part = tail
        match_step = re.search(r"(?:step_|checkpoint-)(\d+)", tail) # 兼容 step_ 和 checkpoint-
        if match_step:
            step = match_step.group(1)
            model_part = os.path.basename(head)
    
    final_model_name = f"{base_model_prefix}-{model_part}"
            
    return {
        "experiment": "lighteval_run", 
        "model": final_model_name, 
        "step": step,
        "dataset": dataset_name
    }

def log_result_to_csv(log_file: str, file_info: Dict, metric_name: str, metric_value: float):
    """将结果（包含step）记录到指定的CSV文件中。"""
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(log_file) == 0:
                writer.writerow(['timestamp', 'experiment', 'model', 'step', 'dataset', 'metric_name', 'metric_value'])
            
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'), 
                file_info['experiment'], 
                file_info['model'], 
                file_info['step'],
                file_info['dataset'], 
                metric_name, 
                f"{metric_value:.4f}"
            ])
    except Exception as e:
        print(f"  [严重错误] 无法写入日志文件 {log_file}: {e}")

# =================== 核心计算逻辑 (完全重构) ===================

def calculate_and_log_token_stats(data_path: str, tokenizer, args: argparse.Namespace):
    """
    重构此函数以正确处理lighteval的details文件结构。
    文件中的每一行代表一个问题。
    """
    # 结构: responses_by_run[run_index] = [token_len_q1, token_len_q2, ...]
    responses_by_run: DefaultDict[int, List[int]] = defaultdict(list)
    total_token_count = 0
    total_responses = 0
    num_questions = 0

    if not os.path.exists(data_path):
        print(f"  [错误] 文件不存在: {data_path}")
        return

    # === 第一步: 逐行读取文件，每行是一个问题 ===
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        num_questions = len(lines)
        if num_questions == 0:
            print("  [信息] 文件为空，跳过。")
            return

        for line in lines:
            try:
                record = json.loads(line)
                model_responses = record.get('model_response', {}).get('text', [])
                
                if not model_responses or not isinstance(model_responses, list):
                    continue

                # 遍历当前问题的所有回答（轮数）
                for run_index, response_text in enumerate(model_responses):
                    if isinstance(response_text, str):
                        token_ids = tokenizer.encode(response_text, add_special_tokens=False)
                        num_tokens = len(token_ids)
                        
                        responses_by_run[run_index].append(num_tokens)
                        total_token_count += num_tokens
                        total_responses += 1
            except (json.JSONDecodeError, KeyError):
                continue

    if total_responses == 0:
        print(f"  [信息] 文件中没有找到有效的答案字段 ('model_response' -> 'text')，跳过。")
        return

    # === 第二步: 计算指标 ===
    num_runs = len(responses_by_run)
    
    # 指标1: 总平均Token
    average_tokens_all = total_token_count / total_responses if total_responses > 0 else 0

    # 指标2 & 3: 最高/最低平均Token
    highest_avg_token = 0
    lowest_avg_token = float('inf')

    if num_runs > 0:
        run_averages = []
        for i in range(num_runs):
            run_tokens = responses_by_run[i]
            # 确保分母不为0
            run_avg = sum(run_tokens) / len(run_tokens) if run_tokens else 0
            run_averages.append(run_avg)
        
        highest_avg_token = max(run_averages) if run_averages else 0
        lowest_avg_token = min(run_averages) if run_averages else 0
    else: # 如果没有有效的轮次，所有指标都一样
        highest_avg_token = average_tokens_all
        lowest_avg_token = average_tokens_all


    # === 第三步: 打印和记录日志 ===
    print(f"  - 总问题数 (文件行数): {num_questions}")
    print(f"  - 每个问题的回答数 (轮数): {num_runs}")
    print(f"  - 总回答数: {total_responses}")
    print(f"  - 总Token数: {total_token_count}")
    print("-" * 30)
    print(f"  - 1. 平均Token长度 (所有回答): {average_tokens_all:.2f}")
    if num_runs > 1:
        print(f"  - 2. 最高平均Token长度 (各轮平均中的最大值): {highest_avg_token:.2f}")
        print(f"  - 3. 最低平均Token长度 (各轮平均中的最小值): {lowest_avg_token:.2f}")
    
    file_info = parse_info(data_path, args.tokenizer_path, args.base_model, args.is_base_model)
    if file_info:
        log_result_to_csv(args.log_file, file_info, 'avg_token_length', average_tokens_all)
        if num_runs > 1:
            log_result_to_csv(args.log_file, file_info, 'highest_avg_token_length', highest_avg_token)
            log_result_to_csv(args.log_file, file_info, 'lowest_avg_token_length', lowest_avg_token)
        print(f"  📝 摘要结果已记录至: {args.log_file}")
    else:
        print(f"  ⚠️  警告: 未能从文件名 '{os.path.basename(data_path)}' 中解析元数据。")
    print("-" * 60)


# =================== 主程序入口 (无变化) ===================
def main(args):
    try:
        print("正在加载Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=False)
        print(f"✅ Tokenizer from '{args.tokenizer_path}' loaded successfully.")
    except Exception as e:
        print(f"❌ 严重错误: 无法加载Tokenizer。请检查路径 '{args.tokenizer_path}'。错误: {e}")
        return

    files_to_process = []
    if os.path.exists(args.input_dir) and os.path.isdir(args.input_dir):
        for filename in os.listdir(args.input_dir):
            if filename.endswith("_results.jsonl"):
                files_to_process.append(os.path.join(args.input_dir, filename))

    if not files_to_process:
        print(f"在目录 '{args.input_dir}' 中未找到任何 `_results.jsonl` 文件。")
        return

    print(f"找到 {len(files_to_process)} 个文件需要处理。日志将保存到: {args.log_file}")

    if args.clear_log:
        if os.path.exists(args.log_file):
            print(f"正在清空旧的日志文件: {args.log_file}")
            os.remove(args.log_file)
        
    for file_path in tqdm(files_to_process, desc="批量计算Token长度"):
        print(f"\n▶️  正在处理: {os.path.basename(file_path)}")
        calculate_and_log_token_stats(file_path, tokenizer, args)

    print("\n✅ 所有Token长度计算处理完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算.jsonl文件的平均/最高/最低Token长度，并记录日志。")
    
    parser.add_argument('--input_dir', type=str, default='/mnt/data/kw/tom/main_experiments/0907_1.5B_main/step_90/results', help="包含 _results.jsonl 文件的根目录。")
    parser.add_argument('--log_file', type=str, default='/mnt/data/kw/tom/main_experiments/0907_1.5B_main/step_90/results/token_len.csv', help="用于追加摘要结果的【统一】CSV日志文件。")
    parser.add_argument('--tokenizer_path', type=str, default='/mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', help="Hugging Face tokenizer的路径（将从此路径推断模型和step）。")
    parser.add_argument('--clear_log', action='store_true', help="在开始前清空旧的日志文件。")
    
    parser.add_argument('--base_model', type=str, default='DeepSeek-R1-Distill-Qwen-7B', help="基础模型的名称，将作为记录在CSV中模型名称的前缀。")
    parser.add_argument('--is_base_model', action='store_true', help="【重要】如果tokenizer_path指向的是一个base model，请添加此标志。")

    args = parser.parse_args()
    main(args)