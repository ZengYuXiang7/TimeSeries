#!/bin/bash


pred_lens=(96 192 336 720)

for pred in "${pred_lens[@]}"
do

   python "run_train.py" --config "TransformerConfig" --pred_len "$pred"  --revin True 

done

# pred_lens=(96 192 336 720)
# win_sizes=(24 48 64 96)

# for win_size in "${win_sizes[@]}"
# do
#   for pred in "${pred_lens[@]}"
#   do
#     echo ">> Running with win_size=${win_size}, pred_len=${pred}"
#     python "run_train.py" \
#       --config "TransformerConfig" \
#       --pred_len "$pred" \
#       --win_size "$win_size" \
#       --revin True \
#       --logger 'mkw'
#   done
# done
