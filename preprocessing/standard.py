from utils.data_validator import DataValidator as DV

import numpy as np
from functools import partial


class StandardScaler:
    def __init__(self):
        self.mean_vector = None
        self.std_vector = None


    def fit(self, train_data):
        DV.is_valid_data_matrix('train_data', train_data)
        self.mean_vector = np.mean(train_data, axis=0)
        self.std_vector = np.std(train_data, axis=0)


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
        data_scaled = (data - self.mean_vector) / (self.std_vector + 1e-8)
        return data_scaled
