# coding : utf-8
# Author : yuxiang Zeng
import subprocess
import pickle
import numpy as np
from utils.exp_sh import once_experiment, log_message


# 在这里写下超参数探索空间
hyper_dict = {
    'idx': [1e-3, 1e-4]
}

######################################################################################################
# 这里是总执行实验顺序！！！！！！！！
def experiment_run():
    # Baselines()
    # Ablation()
    Our_model()
    return True

def Baselines():
    # once_experiment('TimesNetConfig', hyper_dict)
    # once_experiment('timeLLMConfig', hyper_dict)
    return True


def Ablation():
    hyper_dict = {
        # 'dataset': ['a', 'b', 'c'],
        'ablation': [1, 2, 3],
    }
    once_experiment('TestConfig', hyper_dict, grid_search=1, retrain=1)
    return True


def Our_model(hyper=None):
    once_experiment('FinancialConfig', hyper_dict, grid_search=0)
    return True


if __name__ == "__main__":
    log_message("\nExperiment Start!!!")
    experiment_run()
    log_message("All commands executed successfully.\n")


