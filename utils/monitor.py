# coding : utf-8
# Author : yuxiang Zeng
import numpy as np

class EarlyStopping:
    def __init__(self, args):
        self.args = args
        self.patience = args.patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = 0
        self.best_model = None
        self.best_epoch = None

    def __call__(self, epoch, params, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, params, val_loss)
        elif score < self.best_score + self.delta:
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

    def track_one_epoch(self, epoch, model, error):
        if self.args.classification:
            self.track(epoch, model.state_dict(), -1 * error['Acc'])
        else:
            self.track(epoch, model.state_dict(), error['NMAE'])
