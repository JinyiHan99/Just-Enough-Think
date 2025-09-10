# find /mnt/data/kw/tom/main_experiments/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/details/mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -name "*.parquet" -exec cp -v {} /mnt/data/kw/tom/main_experiments/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/results/ \;
# find /mnt/data/kw/tom/main_experiments/Qwen/Qwen3-8B/details/mnt/data/kw/models/Qwen/Qwen3-8B -name "*.parquet" -exec cp -v {} /mnt/data/kw/tom/main_experiments/Qwen/Qwen3-8B/results/ \;

# for j in {20..100..10}; do
for j in 60; do
# 方案 B: 如果 step 是不连续的、自定义的列表
# for j in 10 20 60 150; do

    # --- 定义源文件和目标目录的路径模板 ---
    # 源文件路径可能会有变化，因为它可能在 huggingface 下的一个随机哈希目录里
    # 所以我们用通配符 * 来匹配那个随机目录
    SOURCE_PATTERN="/mnt/data/kw/hjy/ckp/0909_truncated_1_5B_linear_reward/global_step_${j}/actor/huggingface/*/community|math_500|0.parquet"
    
    # 目标目录路径
    DEST_DIR="/mnt/data/kw/tom/main_experiments/0909_truncated_1_5B_linear_reward/step_${j}"

    # --- 检查目标目录是否存在，如果不存在则创建 ---
    # -p 选项可以确保如果父目录不存在，也会一并创建
    if [ ! -d "$DEST_DIR" ]; then
        echo "目标目录不存在，正在创建: $DEST_DIR"
        mkdir -p "$DEST_DIR"
    fi

    # --- 查找并移动文件 ---
    # 使用一个循环来处理可能由通配符匹配到的多个文件（尽管通常只有一个）
    # 使用 shopt -s nullglob 确保如果没有匹配项，循环不会执行
    shopt -s nullglob
    for source_file in $SOURCE_PATTERN; do
        if [ -f "$source_file" ]; then
            echo "正在移动:"
            echo "  从: $source_file"
            echo "  到: $DEST_DIR/"
            # 使用 mv 命令移动文件, -v 显示过程
            mv -v "$source_file" "$DEST_DIR/"
        fi
    done
    shopt -u nullglob # 关闭 nullglob 选项

done

echo "所有操作完成！"

#!/bin/bash

# ===================================================================
# 请在这里定义你的 step 列表 {j}
# 例如, 如果你只想处理 step_60, 就写 60
# 如果你想处理 60, 70, 80, 就写 {60..80..10}
# ===================================================================
for j in 60; do # <-- 这里只处理 step_60, 你可以按需修改

    echo "======================================================"
    echo "正在处理 Step: ${j}"
    
    # --- 1. 定义源搜索目录和目标目录的路径模板 ---
    
    # 源搜索目录：根据你的 ls 输出，这是正确的路径
    SEARCH_DIR="/mnt/data/kw/tom/main_experiments/0909_truncated_1_5B_linear_reward/step_${j}/details/mnt/data/kw/hjy/ckp/0909_truncated_1_5B_linear_reward/global_step_${j}/actor/huggingface"
    
    # 目标目录：根据你的最新需求，这是正确的路径
    DEST_DIR="/mnt/data/kw/tom/main_experiments/0909_truncated_1_5B_linear_reward/step_${j}/results"

    # --- 2. 检查源搜索目录是否存在，如果不存在则跳过 ---
    if [ ! -d "$SEARCH_DIR" ]; then
        echo "  -> 警告：源搜索目录不存在，跳过 step ${j}"
        echo "     检查路径: $SEARCH_DIR"
        continue
    fi

    # --- 3. 检查目标目录是否存在，如果不存在则创建 ---
    # -p 选项可以确保所有层级的父目录都会被创建
    if [ ! -d "$DEST_DIR" ]; then
        echo "  -> 目标目录不存在，正在创建: $DEST_DIR"
        mkdir -p "$DEST_DIR"
    fi

    echo "  搜索目录: ${SEARCH_DIR}"
    echo "  目标目录: ${DEST_DIR}"
    echo "------------------------------------------------------"

    # --- 4. 查找并复制所有 .parquet 文件 ---
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★  修改点: 将 mv (移动) 命令改为 cp (复制)  ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    find "$SEARCH_DIR" -type f -name "*.parquet" -exec cp -v {} "$DEST_DIR" \;
    
    # 检查 find 是否找到了文件。我们可以通过计算文件数来实现。
    count=$(find "$DEST_DIR" -name "*.parquet" | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  -> 成功复制了 ${count} 个 .parquet 文件。"
    else
        echo "  -> 警告：在源目录中没有找到任何 .parquet 文件。"
    fi
    
done

echo "======================================================"
echo "所有 step 处理完成！"