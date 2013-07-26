import numbers

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_arrays

class SelectiveOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_values="auto", dtype=np.float):
        self.n_values = n_values
        self.dtype = dtype

    def fit(self, X, y=None, rare=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None, rare=None):
        X = check_arrays(X, sparse_format='dense', dtype=np.int)[0]
        if np.any(X < 0):
            raise ValueError("X needs to contain only non-negative integers.")
        n_samples, n_features = X.shape
        if self.n_values == 'auto':
            n_values = np.max(X, axis=0) + 1
        elif isinstance(self.n_values, numbers.Integral):
            n_values = np.empty(n_features, dtype=np.int)
            n_values.fill(self.n_values)
        else:
            try:
                n_values = np.asarray(self.n_values, dtype=int)
            except (ValueError, TypeError):
                raise TypeError("Wrong type for parameter `n_values`. Expected"
                                " 'auto', int or array of ints, got %r"
                                % type(X))
            if n_values.ndim < 1 or n_values.shape[0] != X.shape[1]:
                raise ValueError("Shape mismatch: if n_values is an array,"
                                 " it has to be of shape (n_features,).")
        self.n_values_ = n_values
        n_values = np.hstack([[0], n_values])
        indices = np.cumsum(n_values)
        self.feature_indices_ = indices
        
        column_indices = (X + indices[:-1]).ravel()
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)
        data = np.ones(n_samples * n_features)
        out = sp.coo_matrix((data, (row_indices, column_indices)),
                            shape=(n_samples, indices[-1]),
                            dtype=self.dtype).tocsr()

        if self.n_values == 'auto':
            mask = np.array(out.sum(axis=0)).ravel() != 0
            active_features = np.where(mask)[0]
            out = out[:, active_features]
            self.active_features_ = active_features
        
        return out

    def transform(self, X, rare=None):
        X = check_arrays(X, sparse_format='dense', dtype=np.int)[0]
        if np.any(X < 0):
            raise ValueError("X needs to contain only non-negative integers.")
        n_samples, n_features = X.shape

        indices = self.feature_indices_
        if n_features != indices.shape[0] - 1:
            raise ValueError("X has different shape than during fitting."
                             " Expected %d, got %d."
                             % (indices.shape[0] - 1, n_features))

        n_values_check = np.max(X, axis=0) + 1
        if (n_values_check > self.n_values_).any():
            raise ValueError("Feature out of bounds. Try setting n_values.")

        column_indices = (X + indices[:-1]).ravel()
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)
        data = np.ones(n_samples * n_features)
        out = sp.coo_matrix((data, (row_indices, column_indices)),
                            shape=(n_samples, indices[-1]),
                            dtype=self.dtype).tocsr()
        if self.n_values == 'auto':
            out = out[:, self.active_features_]
        return out
