from configs.MainConfig import OtherConfig
from configs.default_config import *


@dataclass
class TimesNetConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'timesnet'  # 模型类型
    bs: int = 32  # 批大小
    epochs: int = 10  # 训练周期
    d_model: int = 64
    d_ff: int = 32  # 前馈层大小
    dropout: float = 0.1  # Dropout 比例
    enc_in: int = 1  # 输入特征数目
    c_out: int = 1  # 输出特征数目
    embed: str = 'fixed'  # 嵌入类型
    freq: str = 'h'  # 时间序列的频率
    top_k: int = 5  # 高频成分数量
    e_layers: int = 2  # 网络层数
    num_kernels: int = 8
    label_len: int = 24
