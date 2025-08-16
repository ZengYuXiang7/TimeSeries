#!/bin/bash
clear
# 预测长度列表
pred_lens=(96 192 336 720)

# 模型配置列表
exp_names=(TransformerConfig)

# 双重循环
for exp in "${exp_names[@]}"
do
  for len in "${pred_lens[@]}"
  do
    echo "Running with exp_name=$exp, pred_len=$len"
    python run_train.py --exp_name "$exp" --retrain 1 --pred_len "$len" --revin True --logger zyx
  done
done