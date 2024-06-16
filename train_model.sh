#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量
experiment=1
run_name='Experiment'
rounds=5 epochs=150 patience=30 device='cpu'
batch_size=64 learning_rate=0.001 decay=0.001
record=1 program_test=0 verbose=1 classification=0
dimensions="40"
datasets="cpu"
densities="0.80"
py_files="train_model"
models="rnn"
#models="lstm"
#models="attlstm"
#models="lstm mlp"
num_windows=12 num_preds=1


for py_file in $py_files
do
    for dim in $dimensions
    do
        for dataset in $datasets
        do
						for density in $densities
            do
            		for model in $models
                do
                    python ./$py_file.py \
                          --device $device \
                          --logger $run_name \
                          --rounds $rounds \
                          --density $density \
                          --dataset $dataset \
                          --patience $patience \
                          --model $model \
                          --bs $batch_size \
                          --epochs $epochs \
                          --patience $patience \
                          --bs $batch_size \
                          --lr $learning_rate \
                          --decay $decay \
                          --program_test $program_test \
                          --dimension $dim \
                          --experiment $experiment \
                          --record $record \
                          --verbose $verbose \
                          --classification $classification \
                          --num_preds $num_preds \
                          --num_windows $num_windows
#                          --train_size $train_size \

                done
            done
        done
    done
done