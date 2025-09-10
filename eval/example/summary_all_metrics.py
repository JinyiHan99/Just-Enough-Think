import json
import re
import numpy as np
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer
from collections import defaultdict
from tqdm import tqdm
import glob
from typing import Dict, Tuple, Optional, List, DefaultDict

# ==============================================================================
# 模块1: 严格从外部文件导入计算函数
# ==============================================================================
try:
    from eval_lighteval_math import cal_metrics
    from eval_csqa_gpqa import evaluate_results
    # 我们只从 token_len_2 导入 ACC 提取函数
    from token_len_and_acc import extract_accuracy_from_file
    print("✅ 所有必要的评估模块已成功导入。")
except ImportError as e:
    print(f"❌ 严重错误: 无法导入必要的评估模块: {e}")
    exit(1)

# ==============================================================================
# 模块2: 移植自 "参考脚本" 的、正确的 Token 长度计算模块
# ==============================================================================
def calculate_token_stats_from_parquet(data_path: str, tokenizer: PreTrainedTokenizer) -> Optional[Dict[str, float]]:
    """
    (移植自参考脚本) 从 Parquet 文件计算详细的 Token 统计数据。
    """
    responses_by_run: DefaultDict[int, List[int]] = defaultdict(list)
    total_token_count, total_responses = 0, 0

    if not data_path or not os.path.exists(data_path): return None
    try:
        df = pd.read_parquet(data_path)
        if 'model_response' not in df.columns: return None

        for item in df['model_response']:
            if not isinstance(item, dict): continue
            # 正确处理 model_response['text'] 的 numpy array 或 list 结构
            model_responses = item.get('text', [])
            if isinstance(model_responses, np.ndarray):
                model_responses = model_responses.tolist() # 转换为 list
            if not isinstance(model_responses, list): continue
            
            for run_index, response_text in enumerate(model_responses):
                if isinstance(response_text, str):
                    token_ids = tokenizer.encode(response_text, add_special_tokens=False)
                    num_tokens = len(token_ids)
                    responses_by_run[run_index].append(num_tokens)
                    total_token_count += num_tokens
                    total_responses += 1
    except Exception as e:
        # print(f"  [Token警告] 读取或处理 Parquet 文件 {os.path.basename(data_path)} 时出错: {e}")
        return None

    if total_responses == 0: return None
        
    average_tokens_all = total_token_count / total_responses
    stats = {'avg_token_length': average_tokens_all}
    return stats

# ==============================================================================
# 模块3: 主流程 (移植自 "参考脚本" 的健壮逻辑)
# ==============================================================================
def main(args):
    """主函数，递归搜索所有步骤文件夹，并处理其中的文件。"""
    
    step_dirs_paths = glob.glob(os.path.join(args.base_dir, "step_*"))
    step_dirs = sorted([d for d in step_dirs_paths if os.path.isdir(d)], key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
    
    if not step_dirs:
        print(f"在 '{args.base_dir}' 下未找到任何 'step_*' 文件夹。")
        return
    print(f"找到 {len(step_dirs)} 个步骤文件夹进行分析.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=False)
    print(f"✅ Tokenizer 加载成功: {args.tokenizer_path}")

    all_steps_data = []

    for step_dir in tqdm(step_dirs, desc="分析所有步骤"):
        step_name = os.path.basename(step_dir)
        step_results = {"Checkpoint": step_name}

        # --- 文件地图构建 ---
        file_map = defaultdict(dict)
        all_files_in_step = glob.glob(os.path.join(step_dir, '**'), recursive=True)
        
        task_name_map = {
            'commonsenseqa': 'csqa', 'gpqa_diamond_instruct': 'gpqa',
            'gsm8k': 'gsm8k', 'aime24_avg': 'aime24', 'AMC_avg': 'AMC',
            'Olympiad': 'Olympiad'
        }

        # 查找 ACC JSON 文件
        acc_json_files = [p for p in all_files_in_step if 'results_' in os.path.basename(p) and p.endswith('.json')]
        
        # 查找 Parquet 和 JSONL 文件
        for f_path in all_files_in_step:
            match = re.search(r'community\|([^|]+)', os.path.basename(f_path))
            if match:
                extracted_name = match.group(1)
                if extracted_name in task_name_map:
                    canonical_name = task_name_map[extracted_name]
                    if f_path.endswith('_results.jsonl'): file_map[canonical_name]['jsonl'] = f_path
                    elif f_path.endswith('.parquet'): file_map[canonical_name]['parquet'] = f_path
        
        # --- 评估 ---

        # CSQA & GPQA
        for task in ['csqa', 'gpqa']:
            path_jsonl = file_map[task].get('jsonl')
            path_parquet = file_map[task].get('parquet')
            acc, token = 0.0, 0
            
            # 核心修复：增加空值检查
            if path_jsonl:
                summary, _, _ = evaluate_results(path_jsonl)
                acc = float(summary.get('accuracy', '0%').strip('%'))
            
            if path_parquet:
                stats = calculate_token_stats_from_parquet(path_parquet, tokenizer)
                token = stats.get('avg_token_length') if stats else 0

            step_results[f'{task}_acc'], step_results[f'{task}_token'] = acc, token

        # MATH
        math_path = os.path.join(step_dir, 'community|math_500|0_results_with_std.jsonl')
        math_acc, math_token = 0.0, 0
        if os.path.exists(math_path):
            math_acc, math_token = cal_metrics(math_path, "sample", "std", model_path=args.tokenizer_path)
        step_results['math_acc'] = math_acc * 100 if isinstance(math_acc, float) else math_acc
        step_results['math_token'] = math_token

        # GSM8K & AIME24
        merged_accs = {}
        for acc_file in acc_json_files:
            # 这里我们使用您提供的 token_len_2.py 中的 extract_accuracy_from_file
            accs_from_file = extract_accuracy_from_file(acc_file)
            for task_name, (_, acc_value) in accs_from_file.items():
                 # 规范化任务名
                if 'gsm8k' in task_name: merged_accs['gsm8k'] = acc_value
                elif 'aime' in task_name: merged_accs['aime24'] = acc_value
                elif 'AMC' in task_name: merged_accs['AMC'] = acc_value
                elif 'Olympiad' in task_name: merged_accs['Olympiad'] = acc_value
        
        for task in ['gsm8k', 'aime24', 'AMC', 'Olympiad']:
            path_parquet = file_map[task].get('parquet')
            token = 0
            if path_parquet:
                stats = calculate_token_stats_from_parquet(path_parquet, tokenizer)
                token = stats.get('avg_token_length') if stats else 0
            
            acc = merged_accs.get(task, 0.0) * 100
            step_results[f'{task}_acc'], step_results[f'{task}_token'] = acc, token

        tasks_with_counts = {
            'csqa': 1221, 'gpqa': 198, 'math': 500, 'gsm8k': 1319,
            'aime24': 300, 'AMC': 830, 'Olympiad': 675
        }

        # --- 计算平均 ACC ---
        valid_accs = []
        for task in tasks_with_counts.keys():
            acc_key = f'{task}_acc'
            # 只有当 acc 值存在且大于 0 时，才计入平均值计算
            if step_results.get(acc_key, 0) > 0:
                valid_accs.append(step_results[acc_key])
        
        # 避免除以零的错误
        if valid_accs:
            step_results['avg_acc'] = sum(valid_accs) / len(valid_accs)
        else:
            step_results['avg_acc'] = 0.0

        # --- 计算加权平均 Token ---
        weighted_token_sum = 0
        total_weight = 0
        for task, count in tasks_with_counts.items():
            token_key = f'{task}_token'
            # 只有当 token 值存在且大于 0 时，才计入加权平均计算
            if step_results.get(token_key, 0) > 0:
                weighted_token_sum += step_results[token_key] * count
                total_weight += count

        # 避免除以零的错误
        if total_weight > 0:
            step_results['avg_token'] = weighted_token_sum / total_weight
        else:
            step_results['avg_token'] = 0

        all_steps_data.append(step_results)
        

    # --- 格式化与输出 ---
    if not all_steps_data:
        print("\n\n⚠️ 未能从任何文件夹中提取到有效数据。")
        return

    df = pd.DataFrame(all_steps_data)
    df.rename(columns={'Checkpoint': 'step'}, inplace=True)
    df = df.set_index("step").sort_index()
    
    formatters = {}
    for col in df.columns:
        if col.endswith('_acc'): formatters[col] = '{:.2f}'.format
        elif col.endswith('_token'): formatters[col] = '{:.0f}'.format
            
    df_formatted = df.fillna(0) # 将 N/A 替换为 0
    
    print("\n" + "="*120)
    print(" " * 50 + "最终评估总结报告")
    print("="*120)
    print(df_formatted.to_string(formatters=formatters))
    print("="*120)

    try:
        df_formatted.to_csv(args.output_csv)
        print(f"\n✅ 结果已成功保存到: {args.output_csv}")
    except Exception as e:
        print(f"\n❌ 保存 CSV 文件时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析ckp")
    parser.add_argument("--base_dir", default='/mnt/data/kw/tom/main_experiments/0906_easyr1', type=str, help="实验的根目录。")
    parser.add_argument("--tokenizer_path", default='/mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', type=str, help="HuggingFace tokenizer 的路径。")
    parser.add_argument("--output_csv", type=str, default='/mnt/data/kw/tom/main_experiments/0906_easyr1/summary_all.csv', help="输出 CSV 文件的路径。")
    args = parser.parse_args()
    if args.output_csv is None:
        args.output_csv = os.path.join(args.base_dir, "_summary_results.csv")
    main(args)