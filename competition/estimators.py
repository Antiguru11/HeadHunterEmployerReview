from abc import ABCMeta, abstractmethod

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer, )
from sklearn.base import (BaseEstimator,
                          clone,
                          MetaEstimatorMixin, 
                          ClassifierMixin,
                          TransformerMixin, )
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
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


class _FromPandasDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, preprocessor) -> None:
        super().__init__()

        self.tokens = preprocessor(X, )

        if y.ndim == 1:
            self.labels = y.tolist()
        elif y.ndim == 2:
            self.labels = y.apply(lambda x: x.tolist(), axis=1).tolist()
        else:
            RuntimeError()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: value[idx] for key, value in self.tokens.items()}
        item['labels'] = self.labels[idx]
        return item


class _MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


class PreTrainedTransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 task_type: str,
                 pretrained_model_name_or_path: str,
                 tokenizer_args: dict,
                 model_args: dict,  ) -> None:
        super().__init__()
        self.task_type = task_type
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer_args = tokenizer_args
        self.model_args = model_args

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name_or_path,
                                                                        **self.model_args)

    def _preprocess_input(self, X):
        sentences = X.iloc[:, 0].tolist()
        return self.tokenizer(sentences, **self.tokenizer_args)

    def fit(self, X, y, **fit_params):
        dataset = _FromPandasDataset(X, y, self._preprocess_input)

        args = TrainingArguments(**fit_params.get('trainings_args', {}))

        trainer = None
        if self.task_type == 'binary' or self.task_type == 'multiclass':
            raise NotImplementedError
        elif self.task_type == 'multilabel':
            trainer = _MultilabelTrainer(self.model,
                                         args,
                                         train_dataset=dataset,
                                         eval_dataset=dataset,
                                         **fit_params.get('trainer_args', {}), )
        trainer.train()
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        if self.task_type == 'binary' or self.task_type == 'multiclass': 
            return (self.predict_proba(X, **kwargs)[:, 1] > 0.5).astype(int)
        elif self.task_type == 'multilabel':
            return (self.predict_proba(X, **kwargs) > 0.5).astype(int)
        else:
            raise RuntimeError()

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        input = self._preprocess_input(X).to('cpu')
        output = self.model.to('cpu')(**input)

        logits = output[0]
        if self.task_type == 'binary' or self.task_type == 'multiclass':
            proba = torch.softmax(logits)
        elif self.task_type == 'multilabel':
            proba = torch.sigmoid(logits)
        else:
            raise RuntimeError()

        return proba.cpu().detach().numpy()
