from utils.data_validator import DataValidator as DV

import numpy as np
from functools import partial
from sklearn.decomposition import TruncatedSVD


class PCA:
    def __init__(self, n_components, solver="randomized"):
        self.n_components = n_components
        self.principal_components = None

        DV.is_supported_function("PCA_solver", solver, self)
        self.solver = getattr(self, solver)


    def full(self, data_matrix):
        U,S,Vh = np.linalg.svd(data_matrix, full_matrices=False)

        if self.n_components is None:
            k = data_matrix.shape[1]
        elif isinstance(self.n_components, int):
            DV.is_positive("Top k components (integer)", self.n_components)
            k = self.n_components

        elif isinstance(self.n_components, float):
            DV.is_in_range(f"Top n% explained variance (float)", self.n_components, 0.0, 1.0)
            explained_variance = S**2
            total_variance = np.sum(explained_variance)
            cum_variance_ratio = np.cumsum(explained_variance / total_variance)
            k = np.argmax(cum_variance_ratio >= self.n_components) + 1

        else:
            raise ValueError("n_components must be an int, a float between 0.0 and 1.0, or None")

        principal_components = Vh.T[:, :k]
        self.principal_components = principal_components
        return principal_components


    def randomized(self, data_matrix):
        if isinstance(self.n_components, int):
            DV.is_positive("Top k components (integer)", self.n_components)
            k = self.n_components
        else:
            k = min(5000, data_matrix.shape[1])

        svd = TruncatedSVD(n_components=k)
        svd.fit(data_matrix)
        principal_components = svd.components_
        
        if isinstance(self.n_components, float):
            DV.is_in_range(f"Top n% explained variance (float)", self.n_components, 0.0, 1.0)
            cum_variance_ratio = np.cumsum(svd.explained_variance_ratio_)
            k_final = np.argmax(cum_variance_ratio >= self.n_components) + 1
            principal_components = svd.components_[:k_final]
        
        self.principal_components = principal_components
        return principal_components


    def fit(self, train_data):
        DV.validate_parameters({
            'train_data': (train_data, (DV.is_valid_data_matrix,))
        })
        self.solver(train_data)


    def fit_transform(self, train_data):
        self.fit(train_data)
        return self.transform(train_data)
    

    def transform(self, test_data):
        DV.validate_parameters({
            'transform': (self.principal_components.all(), ( 
                DV.hath_been_fitted,
            )),
            'transform_data': (test_data, ( 
                DV.is_valid_data_matrix,
                partial(DV.contains_n_columns, n=self.principal_components.shape[1])
            ))
        })
        score_vector = test_data @ self.principal_components.T
        return score_vector
    

    def inverse_transform(self, score_vector):
        DV.validate_parameters({
            'transform': (self.principal_components.all(), ( 
                DV.hath_been_fitted,
            )),
            'score_vector': (score_vector, ( 
                DV.is_valid_data_matrix,
                partial(DV.contains_n_columns, n=self.principal_components.shape[1])
            ))
        })
        reconstructed_data = score_vector @ self.principal_components.T
        return reconstructed_data
