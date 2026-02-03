import numpy as np


class DataValidator():
    def __init__(self):
        pass


    @staticmethod
    def validate_parameters(parameters):
        for parameter, (val, conditions) in parameters.items():
            for condition in conditions:
                condition(parameter, val)


    @staticmethod
    def is_int(parameter, val):
        if not isinstance(val, int):
            raise TypeError(f"{parameter} must be an integer")
        

    @staticmethod
    def is_string(parameter, val):
        if not isinstance(val, str):
            raise TypeError(f"{parameter} must be a string")
    

    @staticmethod
    def is_positive(parameter, val):
        if val <= 0:
            raise ValueError(f"{parameter} must be positive")
    

    @staticmethod
    def is_in_range(parameter, val, min_val, max_val):
        if val < min_val or val >= max_val:
            raise ValueError(f"{parameter} must be between {min_val} and {max_val}")


    @staticmethod
    def is_supported_function(parameter, function, scope):
        if not hasattr(scope, function):
            raise AttributeError(f"{parameter} '{function}' is not supported")
        

    @staticmethod
    def is_supported_metric(parameter, metric, supported):
        if metric not in supported:
            raise AttributeError(f"{parameter} '{metric}' is not a supported metric {supported}")
    

    @staticmethod
    def is_numeric_array(parameter, arr):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{parameter} must be a NumPy array")
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"{parameter} must contain only numeric data")
        
    
    @staticmethod
    def is_not_empty(parameter, arr):
        if arr.size == 0:
            raise ValueError(f"{parameter} is empty or incomplete. Must have at least 1 sample (row) and 1 feature (column)")
        

    @staticmethod
    def is_valid_data_matrix(parameter, arr):
        DataValidator.is_numeric_array(parameter, arr)
        DataValidator.is_not_empty(parameter, arr)
        if arr.ndim != 2:
            raise ValueError(f"{parameter} must be a 2-dimensional matrix")
        

    @staticmethod
    def is_valid_label_matrix(parameter, arr):
        DataValidator.is_numeric_array(parameter, arr)
        DataValidator.is_not_empty(parameter, arr)
        if arr.ndim != 1:
            raise ValueError(f"{parameter} must be a 2-dimensional matrix")
        

    @staticmethod
    def contains_n_columns(parameter, arr, n):
        if arr.shape[1] != n:
            raise ValueError(f"{parameter} must have {n} columns")
        

    @staticmethod
    def n_samples_equals_labels(parameter, data):
        samples, labels = data
        n_samples, n_labels = samples.shape[0], labels.shape[0]
        if n_samples != n_labels:
            raise ValueError(f"Number of {parameter} samples and labels must be equal")
        

    @staticmethod
    def hath_been_fitted(function, parameters):
        if parameters == None:
            raise ValueError(f"Model must be fitted before '{function}'")
        