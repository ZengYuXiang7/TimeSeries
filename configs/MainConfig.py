# coding : utf-8
# Author : yuxiang Zeng

from dataclasses import dataclass
@dataclass
class OtherConfig:
    classification: bool = False
    ablation: int = 0
    try_exp: int = -1

    ts_var: int = 1
    input_size: int = 21
    
    multi_dataset: bool = False
    seq_len: int = 96
    pred_len: int = 96














