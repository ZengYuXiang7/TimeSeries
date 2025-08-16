# coding: utf-8
# Author: mkw
# Date: 2025-06-09 18:11
# Description: TransformerLibraryConfig

from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class TransformerLibraryConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'transformer_library'
    bs: int = 32
    d_model: int = 64
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    num_layers: int = 2
    n_heads: int = 4
    revin: bool = True
    dropout: float = 0.1
    factor: int = 10
    freq: str = 'h'
    embed: str = 'timeF'
    d_ff: int = 96
    activation: str = 'gelu'
    e_layers: int = 2
    d_layers: int = 1
    c_out: int = 21
    label_len: int = 48
