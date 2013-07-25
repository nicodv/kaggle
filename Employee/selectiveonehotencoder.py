import warnings
import numbers
import math

import numpy as np
import numpy.ma as ma
from scipy import sparse
from scipy import stats

from .base import BaseEstimator, TransformerMixin
from .utils import check_arrays
from .utils import array2d
from .utils import as_float_array
from .utils import atleast2d_or_csr
from .utils import atleast2d_or_csc
from .utils import safe_asarray
from .utils import warn_if_not_float
from .utils.fixes import unique
from .utils import deprecated

from .utils.multiclass import unique_labels
from .utils.multiclass import type_of_target

from .utils.sparsefuncs import inplace_csr_row_normalize_l1
from .utils.sparsefuncs import inplace_csr_row_normalize_l2
from .utils.sparsefuncs import inplace_csr_column_scale
from .utils.sparsefuncs import mean_variance_axis0
from .externals import six

class SelectiveOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_values="auto", categorical_features="all",
                 dtype=np.float):
        self.n_values = n_values
        self.categorical_features = categorical_features
        self.dtype = dtype

    def fit(self, X, y=None):
        """Fit OneHotEncoder to X.

Parameters
----------
X : array-like, shape=(n_samples, n_feature)
Input array of type int.

Returns
-------
self
"""
        self.fit_transform(X)
        return self

    def _fit_transform(self, X):
        """Assumes X contains only categorical features."""
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
        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()

        if self.n_values == 'auto':
            mask = np.array(out.sum(axis=0)).ravel() != 0
            active_features = np.where(mask)[0]
            out = out[:, active_features]
            self.active_features_ = active_features

        return out

    def fit_transform(self, X, y=None):
        """Fit OneHotEncoder to X, then transform X.

Equivalent to self.fit(X).transform(X), but more convenient and more
efficient. See fit for the parameters, transform for the return value.
"""
        return _transform_selected(X, self._fit_transform,
                                   self.categorical_features, copy=True)

    def _transform(self, X):
        """Asssumes X contains only categorical features."""
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
        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.n_values == 'auto':
            out = out[:, self.active_features_]
        return out

    def transform(self, X):
        """Transform X using one-hot encoding.

Parameters
----------
X : array-like, shape=(n_samples, n_features)
Input array of type int.

Returns
-------
X_out : sparse matrix, dtype=int
Transformed input.
"""
        return _transform_selected(X, self._transform,
                                   self.categorical_features, copy=True)

