import json
import re
import numpy as np
import os

def extract_predicted_answer(model_response_text):
    """
    从模型的文本输出中提取预测答案。
    核心逻辑：查找所有符合特定模式的答案，并选择位置最靠后的那一个，因为它代表模型的最终结论。
    """
    
    # 每个元组是: (优先级, 正则表达式)
    # 优先级 0 是最高的。
    patterns = [
        (0, re.compile(r"\\boxed\{([A-Z])\}")),
        (1, re.compile(r"ANSWER:\s*([A-Z])\b", re.IGNORECASE)),
        (2, re.compile(r"\bOption ([A-Z])\b", re.IGNORECASE)),
        (3, re.compile(r"\b(?:the\s+)?(?:correct|final|best)?\s*(?:answer|choice)\s*(?:is|would be|should be)\s*:?\s*([A-Z])\b", re.IGNORECASE)),
        # 新增规则：专门用于处理 </think> 标签后跟着 "A. text" 的格式
        (4, re.compile(r"</think>\s*([A-Z])\.\s+[a-zA-Z]{2,}"))
    ]

    found_matches = [] # 将存储元组 (index, priority, letter)

    for priority, pattern in patterns:
        for match in pattern.finditer(model_response_text):
            # finditer 提供了匹配对象，其中包含 start() 位置
            try:
                # 提取捕获组中的字母
                letter = match.group(1).strip().upper()
                if letter and 'A' <= letter <= 'Z':
                    # 存储起始位置、优先级和单个字母
                    found_matches.append((match.start(), priority, letter[0]))
            except IndexError:
                continue
    
    if not found_matches:
        return None

    # 核心逻辑：主要按位置（index）排序，选择最后出现的。
    # 如果位置相同（极不可能），优先级高的（数字小）排在后面，作为最终选择。
    found_matches.sort(key=lambda x: (x[0], -x[1]))
    
    # 返回列表中最后一个匹配项的字母
    return found_matches[-1][2]


def evaluate_results(file_path):
    """
    评估 JSONL 文件中的结果。
    返回: 总体统计数据, 所有错误样本的列表, 所有提取失败样本的列表。
    """
    if not os.path.exists(file_path):
        return {"error": f"文件未找到: {file_path}"}, [], []
        
    total_count, correct_count = 0, 0
    scores, incorrect_samples, extraction_failures = [], [], []

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
                
                if predicted_answer_letter is None:
                    scores.append(0)
                    extraction_failures.append({
                        "line_number": i + 1,
                        "question": data.get('doc', {}).get('query', 'N/A').split('\n')[0],
                        "model_output_snippet": model_response_text[-300:]
                    })
                    continue

                is_correct = (predicted_answer_letter == gold_answer_letter)
                
                if is_correct:
                    correct_count += 1
                    scores.append(1)
                else:
                    scores.append(0)
                    incorrect_samples.append({
                        "line_number": i + 1,
                        "question": data.get('doc', {}).get('query', 'N/A').split('\n')[0],
                        "gold_answer": gold_answer_letter,
                        "extracted_answer": predicted_answer_letter,
                        "model_output_snippet": model_response_text[-300:]
                    })
            except Exception as e:
                print(f"警告: 在第 {i + 1} 行处理时发生错误: {e}")
                continue

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    std_dev = np.std(scores) if scores else 0

    results_summary = {
        "total_samples": total_count,
        "correct_samples": correct_count,
        "incorrect_samples_count": len(incorrect_samples),
        "extraction_failures_count": len(extraction_failures),
        "accuracy": f"{accuracy:.2f}%",
        "std": f"{std_dev:.4f}"
    }

    return results_summary, incorrect_samples, extraction_failures

# --- 主程序 ---
if __name__ == "__main__":
    file_path = "/mnt/data/kw/tom/main_experiments/0907_1.5B_main/step_20/results/details_community|gpqa_diamond_instruct|0_2025-09-08T06-50-52.907031_results.jsonl"
    
    log_file_path = os.path.splitext(file_path)[0] + '.log'

    results, incorrects, failures = evaluate_results(file_path)

    # 1. 在控制台打印最终的总体评估结果
    print("--- 总体评估结果 ---")
    if "error" in results:
        print(results["error"])
    else:
        for key, value in results.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    print("----------------------")
    print(f"\n详细错误日志已写入: {log_file_path}")

    # 2. 将详细的错误列表写入日志文件
    try:
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"--- 所有预测错误的样本 ({len(incorrects)}个) ---\n")
            if not incorrects:
                log_file.write("没有预测错误的样本，表现完美！\n")
            else:
                for sample in incorrects:
                    log_file.write(f"行号: {sample['line_number']}\n")
                    log_file.write(f"  问题: {sample['question'][:100]}...\n")
                    log_file.write(f"  正确答案 (Gold): {sample['gold_answer']}\n")
                    log_file.write(f"  提取的错误答案 (Extracted): {sample['extracted_answer']}\n")
                    log_file.write(f"  模型输出片段: ...{''.join(sample['model_output_snippet'].splitlines())}\n")
                    log_file.write("-" * 20 + "\n")
            
            log_file.write("\n" + "="*40 + "\n\n")

            log_file.write(f"--- 所有提取失败的样本 ({len(failures)}个) ---\n")
            if not failures:
                log_file.write("没有提取失败的样本。\n")
            else:
                for fail in failures:
                    log_file.write(f"行号: {fail['line_number']}\n")
                    log_file.write(f"  问题: {fail['question'][:100]}...\n")
                    log_file.write(f"  模型输出片段: ...{''.join(fail['model_output_snippet'].splitlines())}\n")
                    log_file.write("-" * 20 + "\n")
    except Exception as e:
        print(f"\n错误：无法写入日志文件 {log_file_path}。原因: {e}")