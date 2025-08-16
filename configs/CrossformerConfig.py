
from configs.MainConfig import OtherConfig
from configs.default_config import *


@dataclass
class CrossformerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'crossformer'  
    bs: int = 256
    rank: int = 128
    verbose: int = 1
    seg_len: int = 24