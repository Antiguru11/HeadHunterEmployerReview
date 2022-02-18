from abc import ABCMeta, abstractmethod

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from transformers import (AutoTokenizer,
                          AutoModel, )
from sklearn.base import (BaseEstimator,
                          clone,
                          MetaEstimatorMixin, 
                          ClassifierMixin,
                          TransformerMixin, )
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.utils.metaestimators import available_if
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectorMixin
from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer


def _available_if_base_estimator_has(attr):
    """Return a function to check if `base_estimator` or `estimators_` has `attr`.
    Helper for Chain implementations.
    """

    def _check(self):
        return hasattr(self.base_estimator, attr) or all(
            hasattr(est, attr) for est in self.estimators_
        )

    return available_if(_check)


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


class _BaseChain(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, base_estimator, *, order=None, cv=None, random_state=None):
        self.base_estimator = base_estimator
        self.order = order
        self.cv = cv
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, Y, **fit_params):
        """Fit the model to data matrix X and targets Y.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Y : array-like of shape (n_samples, n_classes)
            The target values.
        **fit_params : dict of string -> object
            Parameters passed to the `fit` method of each step.
            .. versionadded:: 0.23
        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        X, Y = self._validate_data(X, Y, multi_output=True, accept_sparse=True, force_all_finite='allow-nan')

        random_state = check_random_state(self.random_state)
        self.order_ = self.order
        if isinstance(self.order_, tuple):
            self.order_ = np.array(self.order_)

        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif isinstance(self.order_, str):
            if self.order_ == "random":
                self.order_ = random_state.permutation(Y.shape[1])
        elif sorted(self.order_) != list(range(Y.shape[1])):
            raise ValueError("invalid order")

        self.estimators_ = [clone(self.base_estimator) for _ in range(Y.shape[1])]

        if self.cv is None:
            Y_pred_chain = Y[:, self.order_]
            if sp.issparse(X):
                X_aug = sp.hstack((X, Y_pred_chain), format="lil")
                X_aug = X_aug.tocsr()
            else:
                X_aug = np.hstack((X, Y_pred_chain))

        elif sp.issparse(X):
            Y_pred_chain = sp.lil_matrix((X.shape[0], Y.shape[1]))
            X_aug = sp.hstack((X, Y_pred_chain), format="lil")

        else:
            Y_pred_chain = np.zeros((X.shape[0], Y.shape[1]))
            X_aug = np.hstack((X, Y_pred_chain))

        del Y_pred_chain

        for chain_idx, estimator in enumerate(self.estimators_):
            y = Y[:, self.order_[chain_idx]]
            estimator.fit(X_aug[:, : (X.shape[1] + chain_idx)], y, **fit_params)
            if self.cv is not None and chain_idx < len(self.estimators_) - 1:
                col_idx = X.shape[1] + chain_idx
                cv_result = cross_val_predict(
                    self.base_estimator, X_aug[:, :col_idx], y=y, cv=self.cv
                )
                if sp.issparse(X_aug):
                    X_aug[:, col_idx] = np.expand_dims(cv_result, 1)
                else:
                    X_aug[:, col_idx] = cv_result

        return self

    def predict(self, X):
        """Predict on the data matrix X using the ClassifierChain model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        Y_pred : array-like of shape (n_samples, n_classes)
            The predicted values.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=True, reset=False, force_all_finite='allow-nan')
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                if chain_idx == 0:
                    X_aug = X
                else:
                    X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_pred = Y_pred_chain[:, inv_order]

        return Y_pred


class ClassifierChain(MetaEstimatorMixin, ClassifierMixin, _BaseChain):
    """A multi-label model that arranges binary classifiers into a chain.
    Each model makes a prediction in the order specified by the chain using
    all of the available features provided to the model plus the predictions
    of models that are earlier in the chain.
    Read more in the :ref:`User Guide <classifierchain>`.
    .. versionadded:: 0.19
    Parameters
    ----------
    base_estimator : estimator
        The base estimator from which the classifier chain is built.
    order : array-like of shape (n_outputs,) or 'random', default=None
        If `None`, the order will be determined by the order of columns in
        the label matrix Y.::
            order = [0, 1, 2, ..., Y.shape[1] - 1]
        The order of the chain can be explicitly set by providing a list of
        integers. For example, for a chain of length 5.::
            order = [1, 3, 2, 4, 0]
        means that the first model in the chain will make predictions for
        column 1 in the Y matrix, the second model will make predictions
        for column 3, etc.
        If order is `random` a random ordering will be used.
    cv : int, cross-validation generator or an iterable, default=None
        Determines whether to use cross validated predictions or true
        labels for the results of previous estimators in the chain.
        Possible inputs for cv are:
        - None, to use true labels when fitting,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
    random_state : int, RandomState instance or None, optional (default=None)
        If ``order='random'``, determines random number generation for the
        chain order.
        In addition, it controls the random seed given at each `base_estimator`
        at each chaining iteration. Thus, it is only used when `base_estimator`
        exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Attributes
    ----------
    classes_ : list
        A list of arrays of length ``len(estimators_)`` containing the
        class labels for each estimator in the chain.
    estimators_ : list
        A list of clones of base_estimator.
    order_ : list
        The order of labels in the classifier chain.
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `base_estimator` exposes such an attribute when fit.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    See Also
    --------
    RegressorChain : Equivalent for regression.
    MultioutputClassifier : Classifies each output independently rather than
        chaining.
    References
    ----------
    Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank, "Classifier
    Chains for Multi-label Classification", 2009.
    Examples
    --------
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.multioutput import ClassifierChain
    >>> X, Y = make_multilabel_classification(
    ...    n_samples=12, n_classes=3, random_state=0
    ... )
    >>> X_train, X_test, Y_train, Y_test = train_test_split(
    ...    X, Y, random_state=0
    ... )
    >>> base_lr = LogisticRegression(solver='lbfgs', random_state=0)
    >>> chain = ClassifierChain(base_lr, order='random', random_state=0)
    >>> chain.fit(X_train, Y_train).predict(X_test)
    array([[1., 1., 0.],
           [1., 0., 0.],
           [0., 1., 0.]])
    >>> chain.predict_proba(X_test)
    array([[0.8387..., 0.9431..., 0.4576...],
           [0.8878..., 0.3684..., 0.2640...],
           [0.0321..., 0.9935..., 0.0625...]])
    """

    def fit(self, X, Y):
        """Fit the model to data matrix X and targets Y.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Y : array-like of shape (n_samples, n_classes)
            The target values.
        Returns
        -------
        self : object
            Class instance.
        """
        super().fit(X, Y)
        self.classes_ = [
            estimator.classes_ for chain_idx, estimator in enumerate(self.estimators_)
        ]
        return self

    @_available_if_base_estimator_has("predict_proba")
    def predict_proba(self, X):
        """Predict probability estimates.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        Y_prob : array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        X = self._validate_data(X, accept_sparse=True, reset=False, force_all_finite='allow-nan')
        Y_prob_chain = np.zeros((X.shape[0], len(self.estimators_)))
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_prob_chain[:, chain_idx] = estimator.predict_proba(X_aug)[:, 1]
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)
        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_prob = Y_prob_chain[:, inv_order]

        return Y_prob

    @_available_if_base_estimator_has("decision_function")
    def decision_function(self, X):
        """Evaluate the decision_function of the models in the chain.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        Y_decision : array-like of shape (n_samples, n_classes)
            Returns the decision function of the sample for each model
            in the chain.
        """
        X = self._validate_data(X, accept_sparse=True, reset=False, force_all_finite='allow-nan')
        Y_decision_chain = np.zeros((X.shape[0], len(self.estimators_)))
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_decision_chain[:, chain_idx] = estimator.decision_function(X_aug)
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_decision = Y_decision_chain[:, inv_order]

        return Y_decision

    def _more_tags(self):
        return {"_skip_test": True, "multioutput_only": True}


class TfidfTransformer(_TfidfVectorizer):
    def fit(self, raw_documents, y=None):
        if isinstance(raw_documents, pd.DataFrame):
            return super().fit(raw_documents.iloc[:, 0], y)
        else:
            return super().fit(raw_documents, y)

    def fit_transform(self, raw_documents, y=None):
        if isinstance(raw_documents, pd.DataFrame):
            return super().fit_transform(raw_documents.iloc[:, 0], y)
        else:
            return super().fit_transform(raw_documents, y)

    def transform(self, raw_documents):
        if isinstance(raw_documents, pd.DataFrame):
            return super().transform(raw_documents.iloc[:, 0])
        else:
            return super().transform(raw_documents)


class HfEmbeddingsTransfomer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 max_seq_length: int,
                 batch_size: int, ) -> None:
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            data = X.iloc[:, 0].tolist()
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                data = X.tolist()
            elif X.ndim == 2:
                data = X[:, 0].tolist()
            else:
                raise RuntimeError
        else:
            raise RuntimeError

        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        model = AutoModel.from_pretrained(self.pretrained_model_name_or_path)
        model.cuda()

        i = 0
        embeddings = list()
        while True:
            torch.cuda.empty_cache()

            t = tokenizer(data[i*self.batch_size:(i+1)*self.batch_size], 
                          padding=True, truncation=True, max_length=self.max_seq_length, 
                          return_tensors='pt', )
            with torch.no_grad():
                model_output = model(**{k: v.to(model.device) for k, v in t.items()})
            batch_embeddings = model_output.last_hidden_state[:, 0, :]
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings)
            batch_embeddings = batch_embeddings.cpu().numpy()

            embeddings.append(batch_embeddings)

            i += 1
            if i * self.batch_size >= len(data):
                break

        return np.vstack(embeddings)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super(SimpleDataset, self).__init__()
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> dict:
        return self.X[idx], self.y[idx]


class MultilabelMlpClassifier(torch.nn.Module,
                              BaseEstimator,
                              ClassifierMixin, ):
    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 layers_size: list,
                 epochs: int, 
                 batch_size: int, 
                 learning_rate: float,
                 verbose_eval: int = 100, ) -> None:
        super(MultilabelMlpClassifier, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.layers_size = layers_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose_eval = verbose_eval

        if len(layers_size) == 0:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(self.input_size, self.n_classes), )
        else:
            sizes = zip([self.input_size] + layers_size,
                        layers_size + [self.n_classes])
            layers = list()
            for i, (input_dim, output_dim) in enumerate(sizes):
                layers.append(torch.nn.Linear(input_dim, output_dim))
                if i < len(self.layers_size):
                    layers.append(torch.nn.ReLU())

            self.mlp = torch.nn.Sequential(*layers)

    def fit(self, X, y, **fit_params):
        X, y = check_X_y(X, y, multi_output=True)

        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.learning_rate, )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        dataset = SimpleDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size)

        size = len(dataset)
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            for batch, (inputs, labels) in enumerate(dataloader):
                pred = self.forward(inputs)
                loss = loss_fn(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % self.verbose_eval == 0:
                    loss, current = loss.item(), batch * len(inputs)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return self

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        with torch.no_grad():
            logits = self.forward(torch.FloatTensor(X))
            probas = torch.sigmoid(logits)
            return probas.cpu().numpy()

    def forward(self, input):
        logits = self.mlp(input)
        return logits
