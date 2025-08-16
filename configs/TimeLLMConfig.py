from configs.MainConfig import OtherConfig
from configs.default_config import *


@dataclass
class TimeLLMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'timellm'
    task_name: str='timeLLM进行时序预测',
    bs: int = 1
    d_ff: int = 32
    top_k: int = 5
    llm_dim: int = 4096
    d_model: int = 64
    patch_len: int = 16
    stride: int = 8
    llm_model: str = "LLAMA"
    llm_layers: int = 6
    prompt_domain: int = 0
    n_heads: int = 8
    enc_in:int = 7
    dropout: float = 0.1
    dataset: str = "weather"
