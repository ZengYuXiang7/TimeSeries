# coding: utf-8
# Author: mkw
# Date: 2025-06-10 15:45
# Description: TransformerConfig
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class Transformer2Config(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'transformer2'
    bs: int = 32
    d_model: int = 64
    epochs: int = 20
    patience: int = 3
    verbose: int = 1
    num_layers: int = 3
    n_heads: int = 8
    revin: bool = True
    dropout: float = 0.1
    amp: bool = False
    diffusion: bool = True
    noise_scale: float = 1.0
    noise_steps: int = 40
    lamda: float = 0.5  # 用于控制扩散损失的权重
