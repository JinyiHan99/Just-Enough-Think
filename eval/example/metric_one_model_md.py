import json
import re
import numpy as np
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer
from collections import defaultdict
from typing import Dict, Optional, List
import glob

# ==============================================================================
# 模块1: 严格从外部文件导入计算函数
# ==============================================================================
try:
    from eval_lighteval_math import cal_metrics
    from eval_csqa_gpqa import evaluate_results
    from token_len_and_acc import extract_accuracy_from_file
    print("✅ 所有必要的评估模块已成功导入。")
except ImportError as e:
    print(f"❌ 严重错误: 无法导入必要的评估模块: {e}")
    exit(1)

# ==============================================================================
# 模块2: Token 长度计算模块
# ==============================================================================
def calculate_token_stats_from_parquet(data_path: str, tokenizer: PreTrainedTokenizer) -> Optional[Dict[str, float]]:
    """从 Parquet 文件计算详细的 Token 统计数据。"""
    if not data_path or not os.path.exists(data_path): return None
    total_token_count, total_responses = 0, 0
    try:
        df = pd.read_parquet(data_path)
        if 'model_response' not in df.columns: return None
        for item in df['model_response']:
            if not isinstance(item, dict): continue
            model_responses = item.get('text', [])
            if isinstance(model_responses, np.ndarray): model_responses = model_responses.tolist()
            if not isinstance(model_responses, list): continue
            for response_text in model_responses:
                if isinstance(response_text, str):
                    token_ids = tokenizer.encode(response_text, add_special_tokens=False)
                    total_token_count += len(token_ids)
                    total_responses += 1
    except Exception: return None
    if total_responses == 0: return None
    return {'avg_token_length': total_token_count / total_responses}

# ==============================================================================
# 模块3: 主流程
# ==============================================================================
def find_file(files: List[str], patterns: List[str]) -> Optional[str]:
    """在文件列表中查找第一个匹配【所有】模式的文件。"""
    for file in files:
        if all(p in file for p in patterns):
            return file
    return None

def main(args):
    """主函数，递归搜索指定的单个模型文件夹，并处理其中的所有结果文件。"""
    if not os.path.isdir(args.model_dir):
        print(f"❌ 错误: 文件夹 '{args.model_dir}' 不存在。"); return
    
    print(f"🚀 开始分析模型文件夹: {args.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=False)
    print(f"✅ Tokenizer 加载成功: {args.tokenizer_path}")

    model_name = os.path.basename(os.path.normpath(args.model_dir))
    model_results = {"Model": model_name}

    all_files = glob.glob(os.path.join(args.model_dir, '**'), recursive=True)
    print(f"🔍 在目录中找到 {len(all_files)} 个文件。")

    file_map = defaultdict(dict)
    task_patterns = {
        'csqa':     {'jsonl': ['commonsenseqa', '_results.jsonl'], 'parquet': ['commonsenseqa', '.parquet']},
        'gpqa':     {'jsonl': ['gpqa_diamond_instruct', '_results.jsonl'], 'parquet': ['gpqa_diamond_instruct', '.parquet']},
        'mmlu_pro': {'jsonl': ['mmlu_pro', '_results.jsonl'], 'parquet': ['mmlu_pro', '.parquet']},
        'math':     {'jsonl': ['math_500', '_results_with_std.jsonl'], 'parquet': ['math_500', '.parquet']},
        'gsm8k':    {'parquet': ['gsm8k', '.parquet']},
        'aime24':   {'parquet': ['aime24_avg', '.parquet']},
        'AMC':      {'parquet': ['AMC_avg', '.parquet']},
        'Olympiad': {'parquet': ['Olympiad', '.parquet']},
    }
    
    for task, patterns in task_patterns.items():
        if 'jsonl' in patterns: file_map[task]['jsonl'] = find_file(all_files, patterns['jsonl'])
        if 'parquet' in patterns: file_map[task]['parquet'] = find_file(all_files, patterns['parquet'])

    acc_json_files = [p for p in all_files if 'results_' in os.path.basename(p) and p.endswith('.json')]
    
    print("\n--- 📊 开始评估 ---")

    # CSQA, GPQA, MMLU_PRO
    for task in ['csqa', 'gpqa', 'mmlu_pro']:
        path_jsonl, path_parquet = file_map[task].get('jsonl'), file_map[task].get('parquet')
        acc, token = 0.0, 0.0
        if path_jsonl:
            print(f"  › 正在评估 {task.upper()} ACC from: {os.path.basename(path_jsonl)}")
            summary, _, _ = evaluate_results(path_jsonl)
            acc = float(summary.get('accuracy', '0%').strip('%'))
        else: print(f"  › 警告: 未找到 {task.upper()} 的 .jsonl 文件进行 ACC 评估。")
        if path_parquet:
            print(f"  › 正在计算 {task.upper()} token from: {os.path.basename(path_parquet)}")
            stats = calculate_token_stats_from_parquet(path_parquet, tokenizer)
            token = stats.get('avg_token_length', 0.0) if stats else 0.0
        model_results[f'{task}_acc'], model_results[f'{task}_token'] = acc, token

    # MATH
    math_path_jsonl, math_path_parquet = file_map['math'].get('jsonl'), file_map['math'].get('parquet')
    math_acc, math_token = 0.0, 0.0
    if math_path_jsonl:
        print(f"  › 正在评估 MATH ACC from: {os.path.basename(math_path_jsonl)}")
        result = cal_metrics(math_path_jsonl, "sample", "std", model_path=args.tokenizer_path)
        if result:
            acc_val, token_val = result
            math_acc = acc_val * 100 if isinstance(acc_val, float) else 0.0
            math_token = token_val
        else: print(f"  › 错误: 'cal_metrics' for MATH did not return a value.")
    else: print(f"  › 警告: 未找到 MATH 的 '..._results_with_std.jsonl' 文件。")
    if math_path_parquet:
        stats = calculate_token_stats_from_parquet(math_path_parquet, tokenizer)
        if stats and stats.get('avg_token_length'):
            print(f"  › 正在从 Parquet 更新 MATH token from: {os.path.basename(math_path_parquet)}")
            math_token = stats.get('avg_token_length', math_token)
    model_results['math_acc'], model_results['math_token'] = math_acc, math_token

    # GSM8K, AIME24, AMC, Olympiad
    merged_accs = {}
    if acc_json_files:
        print(f"  › 正在从 {len(acc_json_files)} 个 JSON 结果文件中提取 ACCs...")
        for acc_file in acc_json_files:
            accs_from_file = extract_accuracy_from_file(acc_file)
            for task_name, (_, acc_value) in accs_from_file.items():
                if 'gsm8k' in task_name: merged_accs['gsm8k'] = acc_value
                elif 'aime' in task_name: merged_accs['aime24'] = acc_value
                elif 'AMC' in task_name: merged_accs['AMC'] = acc_value
                elif 'Olympiad' in task_name: merged_accs['Olympiad'] = acc_value
    for task in ['gsm8k', 'aime24', 'AMC', 'Olympiad']:
        path_parquet = file_map[task].get('parquet')
        token = 0.0
        if path_parquet:
            print(f"  › 正在计算 {task.upper()} token from: {os.path.basename(path_parquet)}")
            stats = calculate_token_stats_from_parquet(path_parquet, tokenizer)
            token = stats.get('avg_token_length', 0.0) if stats else 0.0
        acc = merged_accs.get(task, 0.0) * 100
        model_results[f'{task}_acc'], model_results[f'{task}_token'] = acc, token

    # --- 计算平均值 ---
    tasks_with_counts = { 'csqa': 1221, 'gpqa': 198, 'math': 500, 'gsm8k': 1319, 'aime24': 300, 'AMC': 830, 'Olympiad': 675, 'mmlu_pro': 2000 }
    
    # 【新增】定义域内和域外任务
    in_domain_tasks = ['gsm8k', 'math', 'aime24', 'AMC', 'Olympiad']
    out_of_domain_tasks = ['csqa', 'gpqa', 'mmlu_pro']
    all_tasks = in_domain_tasks + out_of_domain_tasks
    
    # --- 计算 In-Domain 平均值 ---
    id_valid_accs = [model_results.get(f'{task}_acc', 0) for task in in_domain_tasks if model_results.get(f'{task}_acc', 0) > 0]
    model_results['id_avg_acc'] = sum(id_valid_accs) / len(id_valid_accs) if id_valid_accs else 0.0
    
    id_weighted_token_sum, id_total_weight = 0, 0
    for task in in_domain_tasks:
        token_val = model_results.get(f'{task}_token', 0)
        if token_val > 0:
            id_weighted_token_sum += token_val * tasks_with_counts[task]
            id_total_weight += tasks_with_counts[task]
    model_results['id_avg_token'] = id_weighted_token_sum / id_total_weight if id_total_weight > 0 else 0.0

    # --- 计算 Out-of-Domain 平均值 ---
    ood_valid_accs = [model_results.get(f'{task}_acc', 0) for task in out_of_domain_tasks if model_results.get(f'{task}_acc', 0) > 0]
    model_results['ood_avg_acc'] = sum(ood_valid_accs) / len(ood_valid_accs) if ood_valid_accs else 0.0
    
    ood_weighted_token_sum, ood_total_weight = 0, 0
    for task in out_of_domain_tasks:
        token_val = model_results.get(f'{task}_token', 0)
        if token_val > 0:
            ood_weighted_token_sum += token_val * tasks_with_counts[task]
            ood_total_weight += tasks_with_counts[task]
    model_results['ood_avg_token'] = ood_weighted_token_sum / ood_total_weight if ood_total_weight > 0 else 0.0

    # --- 计算总平均值 ---
    total_valid_accs = id_valid_accs + ood_valid_accs
    model_results['avg_acc'] = sum(total_valid_accs) / len(total_valid_accs) if total_valid_accs else 0.0
    total_weighted_token_sum = id_weighted_token_sum + ood_weighted_token_sum
    total_weight = id_total_weight + ood_total_weight
    model_results['avg_token'] = total_weighted_token_sum / total_weight if total_weight > 0 else 0.0
    
    # --- 格式化与输出 ---
    df = pd.DataFrame([model_results])
    df = df.set_index("Model")
    df_for_csv = df.fillna(0)
    
    df_for_display = pd.DataFrame(index=df.index)
    for col in df_for_csv.columns:
        if '_acc' in col: df_for_display[col] = df_for_csv[col].apply('{:.2f}'.format)
        elif '_token' in col: df_for_display[col] = df_for_csv[col].apply('{:.0f}'.format)
        else: df_for_display[col] = df_for_csv[col]

    print("\n" + "="*80)
    print(" " * 30 + "最终评估总结报告")
    print("="*80)
    print(df_for_display.to_markdown())
    print("="*80)

    try:
        df_for_csv.to_csv(args.output_csv)
        print(f"\n✅ 结果已成功保存到: {args.output_csv}")
    except Exception as e:
        print(f"\n❌ 保存 CSV 文件时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析单个模型结果目录的评估指标。")
    parser.add_argument("--model_dir", default='/mnt/data/kw/tom/main_experiments/0907_1.5B_main/step_60', type=str, help="包含单个模型所有评估结果的根目录。")
    parser.add_argument("--tokenizer_path", default='/mnt/data/kw/models/Qwen/QwQ-32B', type=str, help="HuggingFace tokenizer 的路径。")
    parser.add_argument("--output_csv", type=str, default=None, help="输出 CSV 文件的路径。如果未提供，将自动生成。")
    
    args = parser.parse_args()
    
    if args.output_csv is None:
        model_name = os.path.basename(os.path.normpath(args.model_dir))
        args.output_csv = os.path.join(args.model_dir, f"summary_report_{model_name}.csv")
        
    main(args)