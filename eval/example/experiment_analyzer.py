

import json
import argparse
import os
import re
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer
from collections import defaultdict
import glob
import numpy as np # 导入 numpy 来处理 array

# ==============================================================================
# 模块1: ACC 提取函数 (保持不变)
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
# 模块2: Token 长度计算模块 (最终修正)
# ==============================================================================
def calculate_token_stats_from_parquet(data_path: str, tokenizer: PreTrainedTokenizer) -> Optional[float]:
    """
    (V10: 精确处理 model_response['text'] 的 numpy array 结构)
    """
    total_token_count, total_responses = 0, 0
    if not os.path.exists(data_path): return None
    try:
        df = pd.read_parquet(data_path)
        if 'model_response' not in df.columns: return None

        for item in df['model_response']:
            if not isinstance(item, dict): continue
            
            # --- [最终修正] ---
            # 1. 获取 'text' 键，它可能是一个 numpy array
            model_responses_array = item.get('text')
            
            # 2. 检查它是否是 numpy array 并且不为空
            if not isinstance(model_responses_array, np.ndarray) or model_responses_array.size == 0:
                continue
            
            # 3. 遍历 array 中的每个元素
            for response_text in model_responses_array:
            # --- [修正结束] ---
                if isinstance(response_text, str):
                    token_ids = tokenizer.encode(response_text, add_special_tokens=False)
                    total_token_count += len(token_ids)
                    total_responses += 1
    
    except Exception as e:
        print(f"  [Token警告] 读取或处理 Parquet 文件 {os.path.basename(data_path)} 时出错: {e}")
        return None

    if total_responses == 0:
        return None
        
    return total_token_count / total_responses

# ==============================================================================
# 模块3: 主流程 (保持不变)
# ==============================================================================
def main(args):
    step_dirs = sorted([d for d in glob.glob(os.path.join(args.base_dir, "step_*")) if os.path.isdir(d)])
    if not step_dirs: print(f"在 '{args.base_dir}' 下未找到任何 'step_*' 文件夹。"); return
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

        results_files = glob.glob(os.path.join(step_dir, '**', '*.json'), recursive=True)
        step_accuracies = defaultdict(lambda: ("N/A", None))
        results_files = [p for p in results_files if "results" in p and "huggingface" in p]
        if not results_files: print(f"  [调试信息] 在 step_{step} 中未找到任何 .json 结果文件。")
        for file_path in results_files:
            accuracies = extract_accuracy_from_file(file_path)
            for task_name, (metric, acc_value) in accuracies.items(): step_accuracies[task_name] = (metric, acc_value)

        step_token_stats = defaultdict(lambda: None)
        if tokenizer:
            details_files = glob.glob(os.path.join(step_dir, '**', '*.parquet'), recursive=True)
            details_files = [p for p in details_files if "details" in p]
            if not details_files: print(f"  [调试信息] 在 step_{step} 中未找到任何 .parquet 文件。")
            for file_path in details_files:
                match_task = re.search(r"details_[^|]+\|([^|:]+)", os.path.basename(file_path))
                if match_task:
                    task_name = match_task.group(1)
                    avg_len = calculate_token_stats_from_parquet(file_path, tokenizer)
                    if avg_len is not None: step_token_stats[task_name] = avg_len
        
        all_tasks = set(step_accuracies.keys()) | set(step_token_stats.keys())
        if not all_tasks: continue
        step_summary = {"step": step}
        for task in sorted(list(all_tasks)):
            _, acc_value = step_accuracies[task]
            token_len = step_token_stats[task]
            step_summary[f"{task}_acc"] = acc_value
            step_summary[f"{task}_token_len"] = token_len
        all_steps_data.append(step_summary)

    if not all_steps_data: print("\n\n⚠️ 未能从任何文件夹中提取到有效数据。"); return
    df = pd.DataFrame(all_steps_data).set_index("step").sort_index()
    formatters = {}
    for col in df.columns:
        if col.endswith('_acc'):
            df[col] = pd.to_numeric(df[col], errors='coerce') * 100
            formatters[col] = '{:.1f}'.format
        elif col.endswith('_token_len'):
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
    except Exception as e: print(f"\n❌ 保存 CSV 文件时出错: {e}")
    print("\n--- 实验结果汇总 ---")
    print(df_formatted.to_string(index=False))
    print("------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量分析 lighteval 实验根目录，自动合并准确率和Token长度到 CSV 文件。")
    parser.add_argument("--base_dir", type=str, help="实验的根目录, 例如 '/mnt/data/kw/tom/main_experiments/0907_1.5B_main'。")
    parser.add_argument('--output_csv', type=str, default=None, help="输出 CSV 文件的路径。如果未提供，将默认保存在 base_dir 下，名为 'full_summary_results.csv'。")
    parser.add_argument('--tokenizer_path', type=str, default='/mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', help="Tokenizer 的路径。")
    args = parser.parse_args()
    if args.output_csv is None:
        args.output_csv = os.path.join(args.base_dir, "_summary_results.csv")
    main(args)