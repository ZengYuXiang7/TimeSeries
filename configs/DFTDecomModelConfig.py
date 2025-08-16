# coding: utf-8
# Author: mkw
# Date: 2025-06-08 19:03
# Description: DFTDecomModelConfig

from configs.default_config import *
from dataclasses import dataclass

from configs.MainConfig import OtherConfig


@dataclass
class DFTDecomModelConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'dft'
    bs: int = 32
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    num_layers: int = 2
    revin: bool = True
    d_model: int = 64
    kernel_size: int = 25
    individual: bool = True
    top_k: int = 8
