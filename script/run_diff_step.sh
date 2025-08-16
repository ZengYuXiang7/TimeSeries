#!/bin/bash
pred_lens=(96 192 336 720)
noise_steps_arr=(10 20 30 40 50 60 100 200 1000)
lamda=0.7
diffusion_flags=(True)
for len in "${pred_lens[@]}"
do
  for diffusion in "${diffusion_flags[@]}"
  do
    for noise_steps in "${noise_steps_arr[@]}"
    do
      echo "Running with pred_len=$len, diffusion=$diffusion"
      python run_train.py \
        --exp_name "Transformer2Config" \
        --rounds 3 \
        --retrain 1 \
        --pred_len "$len" \
        --revin True \
        --diffusion "$diffusion" \
        --noise_steps "$noise_steps" \
        --lamda "$lamda"
    done
  done
done