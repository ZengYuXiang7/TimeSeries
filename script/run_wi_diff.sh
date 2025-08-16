#!/bin/bash
pred_lens=(96 192 336 720)
diffusion_flags=(True)
noise_steps=100
lamda=1e-4
for len in "${pred_lens[@]}"
do
  for diffusion in "${diffusion_flags[@]}"
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
        --lamda "$lamda" \
        --patience 5
  done
done
