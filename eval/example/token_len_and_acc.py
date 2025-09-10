# 文件名: experiment_analyzer_v8.py

import json
import argparse
import os
import re
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List, DefaultDict
from transformers import AutoTokenizer, PreTrainedTokenizer
from collections import defaultdict
import glob

# ==============================================================================
# 模块1: 准确率 (ACC) 提取函数 (保持不变)
# ==============================================================================
def find_best_accuracy(metrics: dict) -> Tuple[str, Optional[float]]:
    priority_keys = ["pass@k_with_k&n", "gpqa_avg@k_with_k", "avg@k_with_k", "extractive_match", "em_with_normalize_gold&normalize_pred&type_exact_match", "em_with_type_exact_match", "em_with_normalize_gold&normalize_pred", "em", "pass@1", "acc", "accuracy"]
    for key in priority_keys:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return key, metrics[key]
    return "N/A", None

def extract_accuracy_from_file(file_path: str) -> Dict[str, Tuple[str, float]]:
    accuracies = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results_section = data.get("results")
        if not results_section: return {}
        for task_key, metrics in results_section.items():
            if task_key.lower() == 'all' or not isinstance(metrics, dict): continue
            metric_name, accuracy_value = find_best_accuracy(metrics)
            if accuracy_value is not None:
                task_name = task_key.split('|')[1] if '|' in task_key else task_key
                accuracies[task_name] = (metric_name, accuracy_value)
    except Exception as e:
        print(f"  [ACC警告] 无法处理文件 {os.path.basename(file_path)}: {e}")
    return accuracies

# ==============================================================================
# 模块2: Token 长度计算模块 (核心修改：与 V4 脚本逻辑精确对齐)
# ==============================================================================
def calculate_token_stats_from_parquet(data_path: str, tokenizer: PreTrainedTokenizer) -> Optional[Dict[str, float]]:
    """
    从 Parquet 文件计算详细的 Token 统计数据，逻辑与 V4 脚本对齐。
    返回一个包含 avg, highest_avg, lowest_avg 的字典。
    """
    responses_by_run: DefaultDict[int, List[int]] = defaultdict(list)
    total_token_count = 0
    total_responses = 0

    if not os.path.exists(data_path): return None
    try:
        df = pd.read_parquet(data_path)
        if 'model_response' not in df.columns: return None

        for item in df['model_response']:
            if not isinstance(item, dict): continue
            model_responses = item.get('text', [])
            if not isinstance(model_responses, list): continue
            
            for run_index, response_text in enumerate(model_responses):
                if isinstance(response_text, str):
                    token_ids = tokenizer.encode(response_text, add_special_tokens=False)
                    num_tokens = len(token_ids)
                    responses_by_run[run_index].append(num_tokens)
                    total_token_count += num_tokens
                    total_responses += 1
    except Exception as e:
        print(f"  [Token警告] 读取或处理 Parquet 文件 {os.path.basename(data_path)} 时出错: {e}")
        return None

    if total_responses == 0: return None
        
    # --- 计算指标，逻辑与 V4 脚本完全一致 ---
    num_runs = len(responses_by_run)
    average_tokens_all = total_token_count / total_responses
    
    stats = {'avg_token_length': average_tokens_all}
    
    if num_runs > 1:
        run_averages = [sum(tokens) / len(tokens) if tokens else 0 for tokens in responses_by_run.values()]
        stats['highest_avg_token_length'] = max(run_averages) if run_averages else 0
        stats['lowest_avg_token_length'] = min(run_averages) if run_averages else 0
    else:
        stats['highest_avg_token_length'] = average_tokens_all
        stats['lowest_avg_token_length'] = average_tokens_all
        
    return stats

# ==============================================================================
# 模块3: 主流程 (核心修改：处理并合并新的 Token 统计数据)
# ==============================================================================
def main(args):
    """主函数，递归搜索所有步骤文件夹，并处理其中的文件。"""
    
    step_dirs = sorted([d for d in glob.glob(os.path.join(args.base_dir, "step_*")) if os.path.isdir(d)])
    if not step_dirs:
        print(f"在 '{args.base_dir}' 下未找到任何 'step_*' 文件夹。")
        return
    print(f"找到 {len(step_dirs)} 个步骤文件夹进行分析: {[os.path.basename(d) for d in step_dirs]}")

    tokenizer = None
    try:
        tokenizer_path = args.tokenizer_path
        print(f"\n正在加载 Tokenizer: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        print("✅ Tokenizer 加载成功。")
    except Exception as e:
        print(f"❌ 严重错误: 无法加载 Tokenizer '{tokenizer_path}'。Token 长度将无法计算。错误: {e}")

    all_steps_data = []

    for step_dir in tqdm(step_dirs, desc="分析所有步骤"):
        match = re.search(r"step_(\d+)", os.path.basename(step_dir))
        if not match: continue
        step = int(match.group(1))

        print(f"\n▶️  正在处理 Step: {step}")

        # --- 提取 ACC ---
        results_files = glob.glob(os.path.join(step_dir, '**', '*.json'), recursive=True)
        step_accuracies = defaultdict(lambda: ("N/A", None))
        results_files = [p for p in results_files if "results" in p and "huggingface" in p]
        if not results_files: print(f"  [调试信息] 在 step_{step} 中未找到任何 .json 结果文件。")
        for file_path in results_files:
            accuracies = extract_accuracy_from_file(file_path)
            for task_name, (metric, acc_value) in accuracies.items():
                step_accuracies[task_name] = (metric, acc_value)

        # --- 计算 Token 长度 (逻辑对齐) ---
        step_token_stats = defaultdict(lambda: None)
        if tokenizer:
            details_files = glob.glob(os.path.join(step_dir, '**', '*.parquet'), recursive=True)
            details_files = [p for p in details_files if "details" in p]
            if not details_files: print(f"  [调试信息] 在 step_{step} 中未找到任何 .parquet 文件。")
            for file_path in details_files:
                match_task = re.search(r"details_[^|]+\|([^|:]+)", os.path.basename(file_path))
                if match_task:
                    task_name = match_task.group(1)
                    stats = calculate_token_stats_from_parquet(file_path, tokenizer)
                    if stats:
                        step_token_stats[task_name] = stats
        
        # --- 合并数据 ---
        all_tasks = set(step_accuracies.keys()) | set(step_token_stats.keys())
        if not all_tasks: continue

        step_summary = {"step": step}
        for task in sorted(list(all_tasks)):
            _, acc_value = step_accuracies[task]
            token_stats = step_token_stats[task]
            
            step_summary[f"{task}_acc"] = acc_value
            # 如果 token_stats 存在，则解包所有指标
            if token_stats:
                step_summary[f"{task}_avg_len"] = token_stats.get('avg_token_length')
                step_summary[f"{task}_high_avg_len"] = token_stats.get('highest_avg_token_length')
                step_summary[f"{task}_low_avg_len"] = token_stats.get('lowest_avg_token_length')
            else: # 否则全部设为 None
                step_summary[f"{task}_avg_len"] = None
                step_summary[f"{task}_high_avg_len"] = None
                step_summary[f"{task}_low_avg_len"] = None


        all_steps_data.append(step_summary)

    # --- 格式化与输出 ---
    if not all_steps_data:
        print("\n\n⚠️ 未能从任何文件夹中提取到有效数据。")
        return

    df = pd.DataFrame(all_steps_data).set_index("step").sort_index()
    
    # 构建新的格式化规则
    formatters = {}
    for col in df.columns:
        if col.endswith('_acc'):
            df[col] = pd.to_numeric(df[col], errors='coerce') * 100
            formatters[col] = '{:.1f}'.format
        elif col.endswith('_len'): # 匹配所有 token length 相关的列
            df[col] = pd.to_numeric(df[col], errors='coerce')
            formatters[col] = '{:.0f}'.format
            
    df_formatted = df.fillna('N/A')
    for col, formatter in formatters.items():
        df_formatted[col] = df_formatted[col].apply(lambda x: formatter(x) if x != 'N/A' else 'N/A')
    df_formatted.reset_index(inplace=True)

    try:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        df_formatted.to_csv(args.output_csv, index=False)
        print(f"\n✅ 结果已成功保存到: {args.output_csv}")
    except Exception as e:
        print(f"\n❌ 保存 CSV 文件时出错: {e}")

    print("\n--- 实验结果汇总 ---")
    print(df_formatted.to_string()) # 打印完整表格，不截断
    print("------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量分析 lighteval 实验根目录，自动合并准确率和Token长度到 CSV 文件。")
    parser.add_argument("--base_dir", type=str, help="实验的根目录, 例如 '/mnt/data/kw/tom/main_experiments/0906_easyr1'。")
    parser.add_argument('--output_csv', type=str, default=None, help="输出 CSV 文件的路径。如果未提供，将默认保存在 base_dir 下，名为 'full_summary_results.csv'。")
    parser.add_argument('--tokenizer_path', type=str, default='/mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', help="Tokenizer 的路径。")
    
    args = parser.parse_args()
    if args.output_csv is None:
        args.output_csv = os.path.join(args.base_dir, "full_summary_results.csv")
    main(args)