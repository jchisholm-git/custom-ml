from .base import Base
from utils.data_validator import DataValidator as DV

from scipy.spatial.distance import cdist
import numpy as np
from functools import partial


class KNN(Base):
    def __init__(self, k, d="euclidean"):
        DV.validate_parameters({
            'k': (k, (
                DV.is_int, 
                DV.is_positive
            )),
            'distance_function': (d, (
                DV.is_string,
                partial(DV.is_supported_metric, supported=['euclidean', 'manhattan'])
            ))
        })
        self.k = k
        self.distance = d
        self.data = None
        self.labels = None
        self.classes = None
        self.label_to_index = None
        

    def fit(self, train_data, train_labels):
        DV.validate_parameters({
            'train_data': (train_data, (DV.is_valid_data_matrix,)),
            'train_labels': (train_labels, (DV.is_valid_label_matrix,)),
            'training': ((train_data, train_labels), (DV.n_samples_equals_labels,))
        })
        self.classes = np.unique(train_labels)
        self.label_to_index = {label: i for i, label in enumerate(self.classes)}
        integer_labels = np.searchsorted(self.classes, train_labels)

        self.data = train_data
        self.labels = integer_labels


    def predict(self, test_data):
        test_data = self._reshape_and_validate(test_data)

        indices, distances = self._get_k_smallest(test_data)
        label_weights = 1 / (distances+1e-8)
        weighted_counts = self._get_batch_counts(indices, label_weights)
        
        integer_predictions = np.argmax(weighted_counts, axis=1)
        return self.classes[integer_predictions]
        
    
    def predict_probabilities(self, test_data):
        test_data = self._reshape_and_validate(test_data)

        indices, _ = self._get_k_smallest(test_data)
        weights = np.ones_like(indices, dtype=float)
        counts = self._get_batch_counts(indices, weights)
        
        probability_predictions = counts / self.k
        return probability_predictions
    

    def kneighbors(self, test_data):
        test_data = self._reshape_and_validate(test_data)
        indices, distances = self._get_k_smallest(test_data)
        return distances, indices
    

    def _reshape_and_validate(self, data):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        DV.validate_parameters({
            'test_data': (data, (
                DV.is_valid_data_matrix,
                partial(DV.contains_n_columns, n=self.data.shape[1])
            )),
            'predict': (self.data.all(), (
                DV.hath_been_fitted,
            ))
        })
        return data
    

    def _get_k_smallest(self, test_data):
        distances = cdist(test_data, self.data, self.distance)
        k_smallest_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_smallest_distances = np.take_along_axis(distances, k_smallest_indices, axis=1)

        return (k_smallest_indices, k_smallest_distances)
    

    def _get_batch_counts(self, indices, weights):
        num_classes = len(self.classes)
        label_count_matrix = np.zeros((indices.shape[0], num_classes))
        k_smallest_labels = self.labels[indices]
        
        np.add.at(label_count_matrix, (np.arange(indices.shape[0])[:, None], k_smallest_labels), weights)
        return label_count_matrix


    def evaluate(self, test_data, test_labels):
        DV.validate_parameters({
            'train_data': (test_data, (DV.is_valid_data_matrix,)),
            'train_labels': (test_labels, (DV.is_valid_label_matrix,)),
            'testing': ((test_data, test_labels), (DV.n_samples_equals_labels,))
        })
        predictions = self.predict(test_data)
        correct = predictions == test_labels
        return correct.sum() / len(correct) * 100
    