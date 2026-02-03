from abc import ABC, abstractmethod

class Base(ABC):

    @abstractmethod
    def fit(self, data, labels):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def evaluate(self, test_data, test_labels):
        pass
