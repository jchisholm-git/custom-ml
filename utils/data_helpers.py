from utils.data_validator import DataValidator as DV

import numpy as np


def flatten_features(data):
    DV.validate_parameters({
        'Input matrix': (data, (
            DV.is_numeric_array,
            DV.is_not_empty,
        ))
    })
    return data.reshape(data.shape[0], -1)
