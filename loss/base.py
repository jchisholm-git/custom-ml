from abc import ABC, abstractmethod

import numpy as np


class Base(ABC):

    @abstractmethod
    def __call__(self, true_label, pred_label):
        pass

    @abstractmethod
    def gradient(self, true_label, pred_label):
        pass