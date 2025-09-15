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
            model_responses = item.get('text', [])
            if isinstance(model_responses, np.ndarray):
                model_responses = model_responses.tolist()
            if not isinstance(model_responses, list): continue
            
            for run_index, response_text in enumerate(model_responses):
                if isinstance(response_text, str):
                    token_ids = tokenizer.encode(response_text, add_special_tokens=False)
                    num_tokens = len(token_ids)
                    responses_by_run[run_index].append(num_tokens)
                    total_token_count += num_tokens
                    total_responses += 1
    except Exception as e:
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

        file_map = defaultdict(dict)
        all_files_in_step = glob.glob(os.path.join(step_dir, '**'), recursive=True)
        
        task_name_map = {
            'commonsenseqa': 'csqa', 'gpqa_diamond_instruct': 'gpqa',
            'gsm8k': 'gsm8k', 'aime24_avg': 'aime24', 'AMC_avg': 'AMC',
            'Olympiad': 'Olympiad', 'mmlu_pro': 'mmlu_pro'
        }

        acc_json_files = [p for p in all_files_in_step if 'results_' in os.path.basename(p) and p.endswith('.json')]
        
        for f_path in all_files_in_step:
            match = re.search(r'community\|([^|]+)', os.path.basename(f_path))
            if match:
                extracted_name = match.group(1)
                if extracted_name in task_name_map:
                    canonical_name = task_name_map[extracted_name]
                    if f_path.endswith('_results.jsonl'): file_map[canonical_name]['jsonl'] = f_path
                    elif f_path.endswith('.parquet'): file_map[canonical_name]['parquet'] = f_path
        
        for task in ['csqa', 'gpqa', 'mmlu_pro']:
            path_jsonl, path_parquet = file_map[task].get('jsonl'), file_map[task].get('parquet')
            acc, token = 0.0, 0.0
            if path_jsonl:
                summary, _, _ = evaluate_results(path_jsonl)
                acc = float(summary.get('accuracy', '0%').strip('%'))
            if path_parquet:
                stats = calculate_token_stats_from_parquet(path_parquet, tokenizer)
                token = stats.get('avg_token_length', 0.0) if stats else 0.0
            step_results[f'{task}_acc'], step_results[f'{task}_token'] = acc, token

        math_path = os.path.join(step_dir, 'community|math_500|0_results_with_std.jsonl')
        math_acc, math_token = 0.0, 0.0
        if os.path.exists(math_path):
            # 假设 cal_metrics 返回 (acc, token)
            acc_val, token_val = cal_metrics(math_path, "sample", "std", model_path=args.tokenizer_path)
            math_acc = acc_val * 100 if isinstance(acc_val, float) else 0.0
            math_token = token_val if token_val else 0.0
        step_results['math_acc'], step_results['math_token'] = math_acc, math_token

        merged_accs = {}
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
                stats = calculate_token_stats_from_parquet(path_parquet, tokenizer)
                token = stats.get('avg_token_length', 0.0) if stats else 0.0
            acc = merged_accs.get(task, 0.0) * 100
            step_results[f'{task}_acc'], step_results[f'{task}_token'] = acc, token

        tasks_with_counts = { 'csqa': 1221, 'gpqa': 198, 'math': 500, 'gsm8k': 1319, 'aime24': 300, 'AMC': 830, 'Olympiad': 675, 'mmlu_pro': 2000 }
        
        valid_accs = [step_results.get(f'{task}_acc', 0) for task in tasks_with_counts if step_results.get(f'{task}_acc', 0) > 0]
        step_results['avg_acc'] = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0

        weighted_token_sum, total_weight = 0, 0
        for task, count in tasks_with_counts.items():
            token_val = step_results.get(f'{task}_token', 0)
            if token_val > 0:
                weighted_token_sum += token_val * count
                total_weight += count
        step_results['avg_token'] = weighted_token_sum / total_weight if total_weight > 0 else 0.0

        all_steps_data.append(step_results)

    # --- 格式化与输出 (已修改) ---
    if not all_steps_data:
        print("\n\n⚠️ 未能从任何文件夹中提取到有效数据。")
        return

    df = pd.DataFrame(all_steps_data)
    df.rename(columns={'Checkpoint': 'step'}, inplace=True)
    df = df.set_index("step").sort_index()
    
    # 准备一个用于保存到 CSV 的、包含纯数值的 DataFrame
    df_for_csv = df.fillna(0)
    
    # 准备一个用于在终端显示的、格式化为字符串的 DataFrame
    df_for_display = pd.DataFrame(index=df.index)
    for col in df_for_csv.columns:
        if '_acc' in col:
            df_for_display[col] = df_for_csv[col].apply('{:.2f}'.format)
        elif '_token' in col:
            df_for_display[col] = df_for_csv[col].apply('{:.0f}'.format)
        else:
            df_for_display[col] = df_for_csv[col]
            
    print("\n" + "="*80)
    print(" " * 30 + "最终评估总结报告")
    print("="*80)
    # 使用 to_markdown() 打印格式化的表格
    print(df_for_display.to_markdown())
    print("="*80)

    try:
        # 保存到 CSV 的仍然是原始数值，便于后续处理
        df_for_csv.to_csv(args.output_csv)
        print(f"\n✅ 结果已成功保存到: {args.output_csv}")
    except Exception as e:
        print(f"\n❌ 保存 CSV 文件时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析ckp")
    parser.add_argument("--base_dir", default='/mnt/data/kw/tom/main_experiments/0911_7B_no_curr', type=str, help="实验的根目录。")
    parser.add_argument("--tokenizer_path", default='/mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', type=str, help="HuggingFace tokenizer 的路径。")
    parser.add_argument("--output_csv", type=str, default='/mnt/data/kw/tom/main_experiments/0911_7B_no_curr/summary_all.csv', help="输出 CSV 文件的路径。")
    args = parser.parse_args()
    if args.output_csv is None:
        args.output_csv = os.path.join(args.base_dir, "_summary_results.csv")
    main(args)