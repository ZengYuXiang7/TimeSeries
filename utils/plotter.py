# coding : utf-8
# Author : yuxiang Zeng
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.animation as animation
from utils.utils import *
import textwrap


class MetricsPlotter:
    def __init__(self, filename, args):
        self.args = args
        if args.experiment:
            self.fileroot = f'./results/{args.model}/fig/'
            makedir(self.fileroot)
            self.filename = filename
        else:
            self.fileroot = f'./'
            self.filename = 'None'
        self.all_rounds_results = []
        self.one_round_results = collections.defaultdict(list)

    def append_epochs(self, metrics):
        for key, values in metrics.items():
            self.one_round_results[key].append(values)

    def append_round(self):
        self.all_rounds_results.append(self.one_round_results)

    def reset_round(self):
        self.one_round_results = collections.defaultdict(list)

    def record_metric(self, metrics):
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置全局字体为 'Arial Unicode MS'

        num_rounds = len(self.all_rounds_results)
        num_metrics = len(self.all_rounds_results[0])
        fig, axs = plt.subplots(num_rounds, num_metrics, figsize=(num_metrics * 5, num_rounds * 5))

        for I in range(num_rounds):
            for j, (key, values) in enumerate(self.all_rounds_results[I].items()):
                ax = axs[I, j] if num_rounds > 1 else axs[j]
                ax.plot(range(1, len(values) + 1), values, label=key, linestyle='-', marker='o', markersize=3)
                ax.set_title(key)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(f'Round {I + 1} Metric Value')
                ax.legend()

        # 处理要显示的文本，实现自动换行
        wrapper = textwrap.TextWrapper(width=90)  # 每行最多90个字符
        args_text = "\n".join(wrapper.wrap(str(self.args.__dict__)))

        metrics_text = "\n".join(
            [f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}' for key in metrics]
        )

        combined_text = f"{args_text}\n\n{metrics_text}"

        plt.rcParams['font.size'] = 24  # 设置全局字体大小为 24
        plt.figtext(0.5, -0.2, combined_text, ha='center', va='center')
        plt.tight_layout()
        ts = time.asctime().replace(' ', '_').replace(':', '_')
        if self.args.experiment:
            plt.savefig(self.fileroot + self.filename + f'_{ts}.pdf', bbox_inches='tight', pad_inches=0.5)
        else:
            plt.savefig('./None.pdf', bbox_inches='tight', pad_inches=0.5)
