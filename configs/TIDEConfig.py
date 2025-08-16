# 在您的Config文件中添加

from dataclasses import dataclass
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class TIDEConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    # --- 模型标识 ---
    model: str = 'tide'

    # --- 训练超参数 ---
    bs: int = 256
    lr: float = 0.0001
    epochs: int = 200
    patience: int = 20

    # --- 模型结构超参数 ---
    # input_size 将由数据集自动确定
    d_model: int = 512  # 模型内部隐藏维度
    dropout: float = 0.1
    revin: bool = True

    # --- TIDE 特定超参数 ---
    chunk_size: int = 48  # 分块大小，可以尝试调整

    # --- 占位符/默认值 ---
    rank: int = 32  # TIDE不直接使用rank，但框架可能需要
    weight_decay: float = 0.0001