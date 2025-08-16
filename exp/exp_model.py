# coding : utf-8
# Author : Yuxiang Zeng
# 每次开展新实验都改一下这里
from baselines.CrossFormer.Crossformer import Crossformer
from baselines.Linear import Linear
from baselines.Linear2 import Linear2
from baselines.Linear3 import Linear3
from baselines.Linear4 import Linear4
from baselines.Linear5 import Linear5
from baselines.SeasonalTrendModel import SeasonalTrendModel
from baselines.DFTDecomModel import DFTDecomModel
from baselines.Transformer import Transformer
from baselines.Transformer2 import Transformer2
from baselines.TransformerLibrary import TransformerLibrary
from baselines.TimeLLM.TimeLLM import timeLLM
from baselines.TimesNet.TimesNet import TimesNet
from layers.metric.distance import PairwiseLoss
from exp.exp_base import BasicModel
from modules.backbone import Backbone
from baselines.encoder_seq import SeqEncoder
from modules.backbone_single import SingleModel
from modules.ts_model import TimeSeriesModel


class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.input_size = config.input_size
        self.hidden_size = config.rank
        if config.model == 'ours':
            self.model = TimeSeriesModel(self.input_size, config)
        # 2025年05月30日11:45:49 这里只使用了一层的Linear，效果：
        elif config.model == 'mlp':
            self.model = Linear(self.input_size, config)
        # 2025年6月2日16:06:43 两层MLP,效果
        elif config.model == 'mlp2':
            self.model = Linear2(self.input_size, config)
        # 2025年6月2日16:08:17 三层MLP效果
        elif config.model == 'mlp3':
            self.model = Linear3(self.input_size, config)

        elif config.model == 'mlp4':
            self.model = Linear4(self.input_size, config)

        elif config.model == 'mlp5':
            self.model = Linear5(self.input_size, config)

        elif config.model == 'seasonal_trend_model':
            self.model = SeasonalTrendModel(config)

        elif config.model == 'dft':
            self.model = DFTDecomModel(config)

        elif config.model == 'transformer_library':
            self.model = TransformerLibrary(config)

        elif config.model == 'transformer':  # 添加 transformer 支持
            self.model = Transformer(
                input_size=config.input_size,
                d_model=config.d_model,
                revin=config.revin,
                num_heads=config.n_heads,
                num_layers=config.num_layers,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                match_mode=config.match_mode,
                win_size=config.win_size,
                patch_len=config.patch_len,
                device=config.device
            )

        elif config.model == 'transformer2':  # 添加 transformer 支持
            self.model = Transformer2(
                input_size=config.input_size,
                d_model=config.d_model,
                revin=config.revin,
                num_heads=config.n_heads,
                num_layers=config.num_layers,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                diffusion=config.diffusion,
                noise_scale=config.noise_scale,
                noise_steps=config.noise_steps,
            )

        elif config.model in ['rnn', 'lstm', 'gru']:
            self.model = SeqEncoder(
                input_size=self.input_size,
                d_model=self.hidden_size,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                num_layers=config.num_layers,
                seq_method=config.model,
                bidirectional=True
            )
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

        elif config.model == 'financial':
            if config.multi_dataset:
                self.model = Backbone(self.input_size, config)
            else:
                self.model = SingleModel(self.input_size, config)
            self.distance = PairwiseLoss(method=config.dis_method, reduction='mean')
        else:
            raise ValueError(f"Unsupported model type: {config.model}")


        