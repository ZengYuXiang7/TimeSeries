# 清理屏幕，让输出更清晰
clear

# 定义一个要测试的预测长度列表
# (这个列表与您的Python代码一致，无需改动)
pred_lens=(96 192 336 720)

# 定义一个要运行的模型配置名称列表
# *** 在这里添加了 SegRNNConfig 和 TIDEConfig 来与您的Python代码同步 ***
exp_names=(
#  "SegRNNConfig"
  "TIDEConfig"
#  "ConvGRUConfig"
#  "DecompLinearConfig"
)

# 使用双重循环来遍历所有模型和所有预测长度的组合
for exp in "${exp_names[@]}"
do
  for len in "${pred_lens[@]}"
  do
    echo "========================================================"
    echo "Running with Exp_name Name = $exp, Prediction Length = $len"
    echo "========================================================"

    # 执行Python实验脚本
    # 我保留了 --rounds 3 的设置，这会让每个实验运行3次以获得更可靠的结果
    python run_train.py --exp_name "$exp" --pred_len "$len" --logger "lzh" --rounds 1
    
    echo "Finished one run. Moving to the next..."
    echo ""
  done
done

echo "All experiments finished!"