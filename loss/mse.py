from loss.base import Base

import numpy as np


class MSE(Base):
    def __call__(self, true_label, pred_label):
        return np.mean(np.power(true_label - pred_label, 2))
        

    def gradient(self, true_label, pred_label):
        n = true_label.shape[0]
        return (2 / n) * (pred_label - true_label)
    