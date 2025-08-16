from configs.MainConfig import OtherConfig
from configs.default_config import *


@dataclass
class TimeSeriesConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 56
    epochs: int = 200
    loss_func: str = 'MSELoss'  # L1Loss  MSELoss
    patience: int = 45
    verbose: int = 5
    try_exp: int = 1
    dataset: str = 'weather'  # financial  weather
    multi_dataset: bool = False
    scaler_method: str = 'stander'
    # 组件专区
    num_layers: int = 2
    norm_method: str = 'rms'
    ffn_method: str = 'ffn'
    att_method: str = 'self'
    dis_method: str = 'cosine'
    fft: bool = False
    revin: bool = False
    idx: int = 0
    ts_var: int = True