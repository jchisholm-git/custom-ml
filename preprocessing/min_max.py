from utils.data_validator import DataValidator as DV

import numpy as np
from functools import partial


class MinMaxScaler:
    def __init__(self):
        self.min_vector = None
        self.max_vector = None
        self.diff_vector = None


    def fit(self, train_data):
        DV.validate_parameters({
            'train_data': (train_data, (DV.is_valid_data_matrix,))
        })
        self.min_vector = np.min(train_data, axis=0)
        self.max_vector = np.max(train_data, axis=0)
        self.diff_vector = self.max_vector - self.min_vector


    def fit_transform(self, train_data):
        self.fit(train_data)
        return self.transform(train_data)


    def transform(self, data):
        DV.validate_parameters({
            'transform': (self.min_vector.all(), ( 
                DV.hath_been_fitted,
            )),
            'transform_data': (data, ( 
                DV.is_valid_data_matrix,
                partial(DV.contains_n_columns, n=self.min_vector.shape[0])
            ))
        })
        data_scaled = (data - self.min_vector) / (self.diff_vector + 1e-8)
        return data_scaled
