from utils.data_validator import DataValidator as DV

import numpy as np
from functools import partial


class StandardScaler:
    def __init__(self, use_std=True):
        self.mean_vector = None
        self.std_vector = None
        self.use_std = use_std


    def fit(self, train_data):
        DV.is_valid_data_matrix('train_data', train_data)
        self.mean_vector = np.mean(train_data, axis=0)
        self.std_vector = np.std(train_data, axis=0) if self.use_std else None


    def fit_transform(self, train_data):
        self.fit(train_data)
        return self.transform(train_data)


    def transform(self, data):
        DV.validate_parameters({
            'transform': (self.mean_vector.all(), ( 
                DV.hath_been_fitted,
            )),
            'transform_data': (data, ( 
                DV.is_valid_data_matrix,
                partial(DV.contains_n_columns, n=self.mean_vector.shape[0])
            ))
        })
        if self.use_std:
            return (data - self.mean_vector) / (self.std_vector + 1e-8)
        return data - self.mean_vector
