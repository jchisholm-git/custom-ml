from .base import Base
from utils.data_validator import DataValidator as DV
from loss.mse import MSE

import numpy as np
from functools import partial


class LinearRegression(Base):
    def __init__(self, optimizer, loss_fn, penalty=None, alpha=0.01, l1_ratio=0.5):
        DV.validate_parameters({
            'solver': (penalty, (
                DV.is_string,
                partial(DV.is_supported_metric, supported=['l1', 'l1', 'elasticnet'])
            )),
            'alpha': (alpha, (
                DV.is_positive,
            )),
            'l1_ratio': (l1_ratio, (
                partial(DV.is_in_range, min_val=0.0, max_val=1.0),
            ))
        })
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        self.weights = None
        self.bias = None


    def _gradient_descent(self, train_data, train_labels, epochs, batch_size):
        parameters = np.zeros(train_data.shape[1])
        for e in range(self.epochs):
            learning_rate = self.get_learning_rate(e)
            indices = np.random.permutation(train_data.shape[0])
            data_shuffled, labels_shuffled = train_data[indices], train_labels[indices]

            for i in range(0, train_data.shape[0], self.batch_size):
                data_batch = data_shuffled[i : i+self.batch_size]
                label_batch = labels_shuffled[i : i+self.batch_size]

                predictions = data_batch @ parameters
                error = predictions - label_batch
                parameters -= learning_rate * (data_batch.T @ error) / data_batch.shape[0]
        self.parameters = parameters


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
        bias, weights = 0, np.zeros(train_data.shape[1])
        for e in range(self.epochs):
            indices = np.random.permutation(train_data.shape[0])
            data_shuffled, labels_shuffled = train_data[indices], train_labels[indices]

            for i in range(0, train_data.shape[0], self.batch_size):
                data_batch = data_shuffled[i : i+self.batch_size]
                label_batch = labels_shuffled[i : i+self.batch_size]

                predictions = data_batch @ weights + bias
                error_gradient = self.optimizer.gradient(train_labels, predictions)
                


    def predict(self, test_data):
        ones = np.ones((test_data.shape[0], 1))
        test_data_with_bias = np.hstack((ones, test_data))
        DV.validate_parameters({
            'transform': (self.parameters.all(), ( 
                DV.hath_been_fitted,
            )),
            'transform_data': (test_data_with_bias, ( 
                DV.is_valid_data_matrix,
                partial(DV.contains_n_columns, n=self.parameters.shape[0])
            ))
        })
        return test_data_with_bias @ self.parameters


    def evaluate(self, test_data, test_labels):
        DV.validate_parameters({
            'train_data': (test_data, (DV.is_valid_data_matrix,)),
            'train_labels': (test_labels, (DV.is_valid_label_matrix,)),
            'testing': ((test_data, test_labels), (DV.n_samples_equals_labels,))
        })
        prediction = self.predict(test_data)
    