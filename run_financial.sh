#!/bin/bash
#python run_cluster.py

#all_idx=(0 23 28 30 36 37 55 64 66 72 79 120)
#for i in "${all_idx[@]}"


# for i in {0..11674}
# do
#    python run_train.py --exp_name FinancialConfig --idx $i --seq_len 50 --pred_len 90
# done

#python run_service_single.py --seq_len 25 --pred_len 7 --drop 1
#python run_service_single.py --seq_len 30 --pred_len 30 --drop 0
#python run_service_single.py --seq_len 40 --pred_len 60 --drop 0
#python run_service_single.py --seq_len 50 --pred_len 90 --drop 0

# for i in {0..149}
# do
#     python run_train.py --idx $i --seq_len 36 --pred_len 30
# done


# for i in {0..149}
# do
#     python run_train.py --idx $i --seq_len 96 --pred_len 60
# done

# for i in {0..149}
# do
    # python run_train.py --idx $i --seq_len 96 --pred_len 90
# done
# python run_service.py


python run_service.py --drop 1
