from abc import ABC, abstractmethod

class Base(ABC):

    @abstractmethod
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
        self.iterations = 0

    @abstractmethod
    def step(self, gradients):
        pass

