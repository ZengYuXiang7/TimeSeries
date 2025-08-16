
from dataclasses import dataclass
from configs.MainConfig import OtherConfig
from configs.default_config import *


@dataclass
class FinancialConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'financial'
    bs: int = 16
    d_model: int = 32
    epochs: int = 30
    loss_func: str = 'MSELoss'  # L1Loss  MSELoss
    patience: int = 3
    verbose: int = 5
    try_exp: int = 1
    dataset: str = 'financial'  # financial  weather
    scaler_method: str = 'global'
    monitor_metric: str = 'NMAE'

    multi_dataset: bool = True  # False True
    spliter_ratio: str = '6:2:2'

    seq_len: int = 36
    pred_len: int = 7
    input_size: int = 3
    
    # 组件专区
    num_layers: int = 2
    norm_method: str = 'rms'
    ffn_method: str = 'ffn'
    att_method: str = 'self'
    dis_method: str = 'None'
    fft: bool = False
    revin: bool = True
    idx: int = 0

    # start_date: str = '2022-07-13'
    # end_date: str = '2023-07-13'
    start_date: str = '2020-07-13'
    end_date: str = '2025-06-28'

    n_clusters: int = 160

    constraint: bool = True  # 是否使用约束