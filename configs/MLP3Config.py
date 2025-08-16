
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class MLP3Config(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp3'
    bs: int = 32
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    num_layers: int = 2
    d_model: int = 64
    revin: bool = True