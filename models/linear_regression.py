from .base import Base
from utils.data_validator import DataValidator as DV

import numpy as np
from functools import partial


class LinearRegression(Base):
    def __init__(self, loss_fn, alpha=0.01):
        DV.validate_parameters({
            'alpha': (alpha, ( DV.is_positive,)),
        })
        self.loss_fn = loss_fn
        self.alpha = alpha

        self.weights = None
        self.bias = None


    def fit(self, train_data, train_labels, epochs=100, batch_size=32):
        DV.validate_parameters({
            'train_data': (train_data, (DV.is_valid_data_matrix,)),
            'train_labels': (train_labels, (DV.is_valid_label_matrix,)),
            'training': ((train_data, train_labels), (DV.n_samples_equals_labels,)),
            'epochs': (epochs, (
                DV.is_positive,
                DV.is_int                
            )),
            'batch_size': (batch_size, (
                DV.is_positive,
                DV.is_int 
            ))
        })
        bias, weights = 0.0, np.zeros(train_data.shape[1])
        for e in range(epochs):
            indices = np.random.permutation(train_data.shape[0])
            data_shuffled, labels_shuffled = train_data[indices], train_labels[indices]

            for i in range(0, train_data.shape[0], batch_size):
                data_batch = data_shuffled[i : i + batch_size]
                label_batch = labels_shuffled[i : i + batch_size]

                predictions = data_batch @ weights + bias
                error_gradient = self.loss_fn.gradient(label_batch, predictions)

                weights -= self.alpha * (data_batch.T @ error_gradient)
                bias -= self.alpha * np.sum(error_gradient)
        
        self.bias, self.weights = bias, weights
                

    def predict(self, test_data):
        DV.validate_parameters({
            'transform': (self.weights.all(), ( 
                DV.hath_been_fitted,
            )),
            'transform_data': (test_data, ( 
                DV.is_valid_data_matrix,
                partial(DV.contains_n_columns, n=self.weights.shape[0])
            ))
        })
        return test_data @ self.weights + self.bias


    def evaluate(self, test_data, test_labels):
        DV.validate_parameters({
            'train_data': (test_data, (DV.is_valid_data_matrix,)),
            'train_labels': (test_labels, (DV.is_valid_label_matrix,)),
            'testing': ((test_data, test_labels), (DV.n_samples_equals_labels,))
        })
        predictions = self.predict(test_data)
        return self.loss_fn(test_labels, predictions)
    