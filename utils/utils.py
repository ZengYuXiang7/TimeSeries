# coding : utf-8
# Author : yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年3月9日17:47:08）

import os
import time
import random
import nbformat
import platform
import torch as t
import numpy as np

def set_settings(config):
    
    # 合理的自动化可视训练过程，verbose
    config.verbose = int(config.epochs * 0.10)
    
    if config.debug:
        config.rounds = 2
        config.epochs = 1

    if config.classification:
        config.loss_func = 'CrossEntropyLoss'

    # 检查操作系统
    if platform.system() == "Darwin":  # "Darwin" 是 macOS 的系统标识
        config.device = 'cpu' if config.device != 'mps' else 'mps'

    else:
        # 如果不是 macOS，你可以选择默认设置为 CPU 或 GPU
        config.device = "cuda" if t.cuda.is_available() else "cpu"

    return config


# 时间种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


def conduct_statistical_data(x):
    # 数据统计汇总
    print("\n=== 数据长度统计 ===")
    print(f"样本总数: {len(x)}")
    print(f"最小长度: {np.min(x)}")
    print(f"最大长度: {np.max(x)}")
    print(f"平均长度: {np.mean(x):.2f}")
    print(f"中位数长度: {np.median(x)}")
    print(f"标准差: {np.std(x):.2f}")
    print(f"25% 分位数: {np.percentile(x, 25):.2f}")
    print(f"50% 分位数（即中位数）: {np.percentile(x, 50):.2f}")
    print(f"75% 分位数: {np.percentile(x, 75):.2f}")
    print(f"99 百分位数: {np.percentile(x, 99):.2f}")
    print(f"99.9 百分位数: {np.percentile(x, 99.9):.2f}")
    return True


def computer_info():
    def showinfo(tip, info):
        print("{} : {}".format(tip, info))

    showinfo("操作系统及版本信息", platform.platform())
    showinfo('获取系统版本号', platform.version())
    showinfo('获取系统名称', platform.system())
    showinfo('系统位数', platform.architecture())
    showinfo('计算机类型', platform.machine())
    showinfo('计算机名称', platform.node())
    showinfo('处理器类型', platform.processor())
    showinfo('计算机相关信息', platform.uname())


