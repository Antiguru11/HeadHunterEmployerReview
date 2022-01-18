import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin


class BaseSelector(BaseEstimator, SelectorMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = None

    def fit(self, X, y = None, **fit_params):
        raise NotImplementedError
    
    def _get_support_mask(self):
        return self.mask


class DummySelector(BaseSelector):
    def fit(self, X, y=None, **fit_params):
        self.mask = (pd.DataFrame(X).nunique() > 1).values
        return self

    def _more_tags(self):
        return {'allow_nan': True, }

