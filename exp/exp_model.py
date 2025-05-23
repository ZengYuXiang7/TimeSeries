# coding : utf-8
# Author : Yuxiang Zeng
# 每次开展新实验都改一下这里
from layers.metric.distance import PairwiseLoss
from exp.exp_base import BasicModel
from modules.backbone import Backbone
from baselines.TimeLLM import timeLLM
from baselines.encoder_seq import SeqEncoder
from baselines.mlp import MLP
from baselines.cross_former import Crossformer
from baselines.timesnet import TimesNet

class Model(BasicModel):
    def __init__(self, datamodule, config):
        super().__init__(config)
        self.config = config
        # self.input_size = datamodule.train_loader.dataset..shape[-1]
        self.input_size = 3
        self.hidden_size = config.rank

        if config.model == 'ours':
            self.model = Backbone(self.input_size, config)
            self.distance = PairwiseLoss(method=config.dis_method, reduction= 'mean')
        elif config.model == 'mlp':
            self.model = MLP(input_dim=self.input_size * config.seq_len, hidden_dim=self.hidden_size, output_dim=config.pred_len, n_layer=config.num_layers, init_method='xavier')
        elif config.model in ['rnn', 'lstm', 'gru']:
            self.model = SeqEncoder(input_size=self.input_size, d_model=self.hidden_size, seq_len=config.seq_len, pred_len=config.seq_len, num_layers=config.num_layers, seq_method=config.model, bidirectional=True)
        elif config.model == 'crossformer':  # 添加 Crossformer 支持
            self.model = Crossformer(
                data_dim=self.input_size,
                in_len=config.seq_len,
                out_len=config.pred_len,
                seg_len=config.seg_len,  # 使用 config.seg_len
                win_size=4,
                d_model=self.hidden_size,
                n_heads=8,
                e_layers=2,
                dropout=0.1,
                device=config.device
            )
        elif config.model == 'timesnet':
            self.model = TimesNet(enc_in=self.input_size, configs=config)

        elif config.model == 'timellm':
            self.model = timeLLM(config)

        else:
            raise ValueError(f"Unsupported model type: {config.model}")

