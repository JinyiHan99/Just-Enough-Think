# find /mnt/data/kw/tom/main_experiments/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/details/mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -name "*.parquet" -exec cp -v {} /mnt/data/kw/tom/main_experiments/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/results/ \;
# find /mnt/data/kw/tom/main_experiments/Qwen/Qwen3-8B/details/mnt/data/kw/models/Qwen/Qwen3-8B -name "*.parquet" -exec cp -v {} /mnt/data/kw/tom/main_experiments/Qwen/Qwen3-8B/results/ \;


for j in 60; do

    SOURCE_PATTERN="/mnt/data/kw/hjy/ckp/0909_truncated_1_5B_linear_reward/global_step_${j}/actor/huggingface/*/community|math_500|0.parquet"
    

    DEST_DIR="/mnt/data/kw/tom/main_experiments/0909_truncated_1_5B_linear_reward/step_${j}"


    if [ ! -d "$DEST_DIR" ]; then
        echo "目标目录不存在，正在创建: $DEST_DIR"
        mkdir -p "$DEST_DIR"
    fi

    shopt -s nullglob
    for source_file in $SOURCE_PATTERN; do
        if [ -f "$source_file" ]; then
            echo "正在移动:"
            echo "  从: $source_file"
            echo "  到: $DEST_DIR/"
            cp -v "$source_file" "$DEST_DIR/"
        fi
    done
    shopt -u nullglob # 关闭 nullglob 选项

done

echo "所有操作完成！"

for j in 60; do # <-- 这里只处理 step_60, 你可以按需修改

    echo "======================================================"
    echo "正在处理 Step: ${j}"

    SEARCH_DIR="/mnt/data/kw/tom/main_experiments/0909_truncated_1_5B_linear_reward/step_${j}/details/mnt/data/kw/hjy/ckp/0909_truncated_1_5B_linear_reward/global_step_${j}/actor/huggingface"
    
    DEST_DIR="/mnt/data/kw/tom/main_experiments/0909_truncated_1_5B_linear_reward/step_${j}/results"

    if [ ! -d "$SEARCH_DIR" ]; then
        echo "  -> 警告：源搜索目录不存在，跳过 step ${j}"
        echo "     检查路径: $SEARCH_DIR"
        continue
    fi

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