#!/bin/bash
pred_lens=(96 192 336 720)
noise_steps=40
lamda_arr=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
diffusion_flags=(True)
for len in "${pred_lens[@]}"
do
  for diffusion in "${diffusion_flags[@]}"
  do
    for lamda in "${lamda_arr[@]}"
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
