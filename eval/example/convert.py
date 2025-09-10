import pandas as pd
import os
import json

# --- 配置 ---
# 设置包含 Parquet 文件的输入目录
INPUT_DIR = "/mnt/data/kw/tom/data_test/live_code_bench"
# 设置保存 JSONL 文件的输出目录 (我们可以使用同一个目录)
OUTPUT_DIR = '/mnt/data/kw/tom/data_test/live_code_bench'
# --- 结束配置 ---

def convert_parquet_to_jsonl(input_dir, output_dir):
    """
    查找指定目录中的所有 .parquet 文件，并将它们转换为 .jsonl 格式。
    """
    print(f"开始扫描目录: {input_dir}")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 只处理 .parquet 文件
        if filename.endswith(".parquet"):
            input_path = os.path.join(input_dir, filename)
            
            # 构建输出文件名，将 .parquet 替换为 .jsonl
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_results.jsonl"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"  -> 正在转换: {filename}  =>  {output_filename}")
            
            try:
                # 使用 pandas 读取 Parquet 文件
                df = pd.read_parquet(input_path, engine='pyarrow')
                
                # 将 DataFrame 转换为 JSONL 格式并写入文件
                # 'records' 模式会生成一个字典列表，'lines=True' 会让每条记录占一行
                df.to_json(output_path, orient='records', lines=True, force_ascii=False)

            except Exception as e:
                print(f"    [错误] 转换文件 {filename} 时出错: {e}")

    print("\n所有 .parquet 文件转换完成！")


if __name__ == "__main__":
    convert_parquet_to_jsonl(INPUT_DIR, OUTPUT_DIR)