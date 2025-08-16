# coding : utf-8
# Author : yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年3月9日17:47:08）

import numpy as np

class EarlyStopping:
    def __init__(self, config):
        self.config = config
        self.patience = config.patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 1e9
        self.delta = 0
        self.best_model = None
        self.best_epoch = None

    def __call__(self, epoch, params, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, params, val_loss)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_epoch = epoch
            self.best_score = score
            self.save_checkpoint(epoch, params, val_loss)
            self.counter = 0

    def track(self, epoch, params, error):
        self.__call__(epoch, params, error)

    def save_checkpoint(self, epoch, params, val_loss):
        self.best_epoch = epoch + 1
        self.best_model = params
        self.val_loss_min = val_loss

    def early_stop(self):
        return self.counter >= self.patience

    def track_one_epoch(self, epoch, model, error, metric):
        if self.config.classification:
            self.track(epoch, model.state_dict(), -1 * error[metric])
        else:
            self.track(epoch, model.state_dict(), error[metric])
