import os
from abc import ABC, abstractmethod
from inspect import isclass, isfunction
from functools import partial

import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .utils import get_object


def _dummy_transform(input):
    return input


class PipelineBase(ABC):
    name = None

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    @abstractmethod
    def fit(self, data: pd.DataFrame, config: dict) -> dict:
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def load(path: str):
        pass

    def get_args(self, config: dict) -> dict:
        input = config

        if isinstance(input, dict):
            if (('class' in input or 'type' in input)
                and 'args' in input
                and isinstance(input['args'], dict)):
                return self.get_estimator(input)
            else:
                args = dict()
                for k, v in input.items():
                    args[k] = self.get_args(v)
                return args
        elif isinstance(input, list):
            args = list()
            for v in input:
                args.append(self.get_args(v))
            return args
        else:
            return input
    
    def get_estimator(self, config: dict):
        type_name = 'class' if 'class' in config else 'type'

        obj = get_object(config[type_name])
        args = self.get_args(config.get('args', {}))

        if isclass(obj):
            return obj(**args)
        elif isfunction(obj):
            if config['callback']:
                return obj(**args)
            else:
                return partial(obj, **args)
        else:
            raise ValueError

    def get_pipeline(self, config: dict) -> Pipeline:
        steps = list()

        if 'transformers' in config:
            t_steps = list()
            for t_config in config['transformers']:
                e_steps = [(e['name'], self.get_estimator(e))
                           for e in t_config['estimators']]
                t_steps.append((t_config['name'], Pipeline(e_steps), t_config['columns']))
            t_steps.append(('filter', 
                            FunctionTransformer(func=_dummy_transform),
                            make_column_selector(dtype_exclude=['object', 'datetime'])))
            steps.append(('transformers', ColumnTransformer(t_steps, remainder='drop')))
        
        if 'selectors' in config:
            s_steps = [(e['name'], self.get_estimator(e))
                       for e in config['selectors']]
            steps.append(('selectors', Pipeline(s_steps)))
        
        if 'model' not in config:
            raise RuntimeError
        steps.append(('model', self.get_estimator(config['model'])))

        return Pipeline(steps, verbose=self.verbose)


class SimplePipeline(PipelineBase):
    name = 'simple'

    def __init__(self, verbose: bool) -> None:
        super().__init__(verbose)
        self.features = None
        self.pipeline = None

    def fit(self, data: pd.DataFrame, config: dict) -> dict:
        self.features = config['features']

        X = data.loc[:, self.features]
        Y = pd.DataFrame(index=data.index, columns=range(0, 9))
        for i in range(0, 9):
            Y[i] = data['target'].str.contains(str(i)).astype(int)

        eval_splitter = self.get_estimator(config['evaluation']['split'])
        eval_metrics = {m['name']: self.get_estimator(m) 
                        for m in config['evaluation']['metrics']}

        self.pipeline = self.get_pipeline(config)

        train_idx, test_idx = next(eval_splitter.split(X, Y))

        self.pipeline.fit(X.iloc[train_idx, :],
                          Y.iloc[train_idx, :],
                          **self.get_args(config['fit_params']))

        Y_pred = pd.DataFrame(self.pipeline.predict(X.iloc[test_idx, :]),
                              index=test_idx, columns=range(0, 9))
            
        results = {}
        results['metric_val'] = {n: f(Y.iloc[test_idx, ], Y_pred) 
                                 for n, f in eval_metrics.items()}
        return results

    def predict(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X = data.loc[:, self.features]

        proba = self.pipeline.predict_proba(X)

        if isinstance(proba, list):
            proba = np.vstack([p[:, 1] for p in proba]).T
        
        Y_proba = pd.DataFrame(proba, index=data.index, columns=list(range(0, 9)))

        Y_pred = (Y_proba > 0.5).astype(int)
        Y_pred['target'] = Y_pred.apply(lambda x: ','.join([str(i) 
                                                            for i in range(0, 9) 
                                                            if x[i] == 1]),
                                        axis=1)
        
        mask = (Y_proba <= 0.5).all(axis=1)
        Y_pred.loc[mask, 'target'] = Y_proba.loc[mask, :].idxmax(axis=1).astype(str)

        Y_pred['review_id'] = data.loc[:, 'review_id']
        return Y_pred.loc[:, ['review_id', 'target']]


    def save(self, path: str) -> None:
        with open(os.path.join(path, 'pipeline.pickle'), 'wb') as file:
            pickle.dump(self.pipeline, file)
        with open(os.path.join(path, 'features.txt'), 'w') as file:
            for f in self.features:
                file.write(f + "\n")

    @staticmethod
    def load(path: str):
        obj = SimplePipeline(verbose=False)
        with open(os.path.join(path, 'pipeline.pickle'), 'rb') as file:
            obj.pipeline = pickle.load(file)
        with open(os.path.join(path, 'features.txt'), 'r') as file:
            obj.features = list(map(str.strip, file.readlines()))
        return obj


class BinaryOvaPipeline(PipelineBase):
    name = 'binary_ova'

    def __init__(self, verbose: bool) -> None:
        super().__init__(verbose)
        self.features = None
        self.pipeline_binary = None
        self.pipeline_ova = None

    def _fit_pipeline(self,
                      X: pd.DataFrame,
                      Y: pd.DataFrame,
                      config: dict) -> tuple[Pipeline, dict]:
        eval_splitter = self.get_estimator(config['evaluation']['split'])
        eval_metrics = {m['name']: self.get_estimator(m) 
                        for m in config['evaluation']['metrics']}

        pipeline = self.get_pipeline(config)

        train_idx, test_idx = next(eval_splitter.split(X, Y))

        results = {}
        if len(Y.shape) == 1:
            pipeline.fit(X.iloc[train_idx, :], Y.iloc[train_idx])

            y_pred = pipeline.predict(X.iloc[test_idx, :])
            
            results['metric_val'] = {n: f(Y.iloc[test_idx], y_pred) 
                                    for n, f in eval_metrics.items()}
        elif len(Y.shape) == 2:
            pipeline.fit(X.iloc[train_idx, :], Y.iloc[train_idx, :])

            Y_pred = pipeline.predict(X.iloc[test_idx, :])
            
            results['metric_val'] = {n: f(Y.iloc[test_idx, :], Y_pred) 
                                    for n, f in eval_metrics.items()}
        else:
            raise RuntimeError('Target have unvalid shape')

        return pipeline, results

    def fit(self, data: pd.DataFrame, config: dict) -> dict:
        self.features = config['features']

        X = data.loc[:, self.features]
        Y = pd.DataFrame(index=data.index, columns=range(0, 9))
        for i in range(0, 9):
            Y[i] = data['target'].str.contains(str(i)).astype(int)

        eval_splitter = self.get_estimator(config['evaluation']['split'])
        eval_metrics = {m['name']: self.get_estimator(m) 
                        for m in config['evaluation']['metrics']}

        train_idx, test_idx = next(eval_splitter.split(X, Y))

        self.pipeline_binary, binary_results = self._fit_pipeline(X.iloc[train_idx, :],
                                                                  Y.iloc[train_idx, 0],
                                                                  config['binary'], )
        binary_preds = self.pipeline_binary.predict(X.iloc[test_idx, :])

        ova_train_mask = Y.iloc[train_idx, 0] == 0
        ova_test_mask = Y.iloc[test_idx, 0] == 0
        self.pipeline_ova, ova_results = self._fit_pipeline(X.iloc[train_idx, :][ova_train_mask],
                                                            Y.iloc[train_idx, 1:][ova_train_mask],
                                                            config['ova'],)
        ova_preds = self.pipeline_ova.predict(X.iloc[test_idx, :][ova_test_mask])

        Y_pred = pd.DataFrame(index=test_idx, columns=range(0, 9))
        Y_pred.loc[:, 0] = binary_preds
        Y_pred.loc[ova_test_mask, list(range(1, 9))] = ova_preds
        Y_pred.fillna(0, inplace=True)
            
        results = {}
        results['binary'] = binary_results
        results['ova'] = ova_results
        results['metric_val'] = {n: f(Y.iloc[test_idx, :], Y_pred) 
                                 for n, f in eval_metrics.items()}
        return results

    def predict(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X = data.loc[:, self.features]

        proba = self.pipeline_ova.predict_proba(X)
        if isinstance(proba, list):
            proba = np.vstack([p[:, 1] for p in proba])
        proba = np.vstack([self.pipeline_binary.predict_proba(X)[:, 1], proba]).T

        Y_proba = pd.DataFrame(proba, index=data.index, columns=range(0, 9))
        
        Y_pred = (Y_proba > 0.5).astype(int)
        Y_pred['target'] = Y_pred.apply(lambda x: ','.join([str(i) 
                                                            for i in range(0, 9) 
                                                            if x[i] == 1]),
                                        axis=1)
        
        mask = (Y_proba <= 0.5).all(axis=1)
        Y_pred.loc[mask, 'target'] = (Y_proba
                                      .loc[mask, :]
                                      .idxmax(axis=1).astype(str))

        Y_pred['review_id'] = data.loc[:, 'review_id']
        return Y_pred.loc[:, ['review_id', 'target']]


    def save(self, path: str) -> None:
        with open(os.path.join(path, 'features.txt'), 'w') as file:
            for f in self.features:
                file.write(f + "\n")
        with open(os.path.join(path, 'pipeline_binary.pickle'), 'wb') as file:
            pickle.dump(self.pipeline_binary, file)
        with open(os.path.join(path, 'pipeline_ova.pickle'), 'wb') as file:
            pickle.dump(self.pipeline_ova, file)

    @staticmethod
    def load(path: str):
        obj = BinaryOvaPipeline()
        with open(os.path.join(path, 'features.txt'), 'r') as file:
            obj.features = list(map(str.strip, file.readlines()))
        with open(os.path.join(path, 'pipeline_binary.pickle'), 'rb') as file:
            obj.pipeline_binary = pickle.load(file)
        with open(os.path.join(path, 'pipeline_ova.pickle'), 'rb') as file:
            obj.pipeline_ova = pickle.load(file)
        return obj


pipelines = {SimplePipeline.name: SimplePipeline,
             BinaryOvaPipeline.name: BinaryOvaPipeline, }
