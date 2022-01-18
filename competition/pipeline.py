import os
from abc import ABC, abstractmethod
from inspect import isclass, isfunction
from functools import partial

import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .utils import get_object


class PipelineBase(ABC):
    name = None

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

    @staticmethod
    def get_args(config: dict) -> dict:
        args = dict(config)
        for key, value in args.items():
            if isinstance(value, dict):
                if (('class' in value or 'type' in value)
                    and 'args' in value
                    and isinstance(value['args'], dict)):
                    args[key] = PipelineBase.get_estimator(value)
                else:
                    args[key] = PipelineBase.get_args(value)
            elif isinstance(value, list):
                args[key] = [PipelineBase.get_args(v) for v in value]
            else: 
                pass
        return args
    
    @staticmethod
    def get_estimator(config: dict):
        type_name = 'class' if 'class' in config else 'type'

        obj = get_object(config[type_name])
        args = PipelineBase.get_args(config.get('args', {}))

        if isclass(obj):
            return obj(**args)
        elif isfunction(obj):
            return partial(obj, **args)
        else:
            raise ValueError

    @staticmethod
    def get_pipeline(config: dict) -> Pipeline:
        steps = list()

        if 'transformers' in config:
            t_steps = list()
            for t_config in config['transformers']:
                e_steps = [(e['name'], PipelineBase.get_estimator(e))
                           for e in t_config['estimators']]
                t_steps.append((t_config['name'], Pipeline(e_steps), t_config['columns']))
            steps.append(('transformers', ColumnTransformer(t_steps, remainder='passthrough')))
        
        if 'selectors' in config:
            s_steps = [(e['name'], PipelineBase.get_estimator(e))
                       for e in config['selectors']]
            steps.append(('selectots', Pipeline(s_steps)))
        
        if 'model' not in config:
            raise RuntimeError
        steps.append(('model', PipelineBase.get_estimator(config['model'])))

        return Pipeline(steps)


class SimpleOVAPipeline(PipelineBase):
    name = 'simple_ova'

    def __init__(self) -> None:
        super().__init__()
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

        self.pipeline.fit(X.loc[train_idx, :], Y.loc[train_idx, :])

        Y_pred = pd.DataFrame(self.pipeline.predict(X.loc[test_idx, :]),
                              index=test_idx, columns=range(0, 9))
            
        results = {}
        results['metric_val'] = {n: f(Y.loc[test_idx, ], Y_pred) 
                                 for n, f in eval_metrics.items()}
        return results

    def predict(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X = data.loc[:, self.features]
        Y_pred = pd.DataFrame(self.pipeline.predict(X),
                              index=data.index, columns=range(0, 9))
        Y_pred['review_id'] = data.loc[:, 'review_id']
        Y_pred['target'] = Y_pred.apply(lambda x: ','.join([str(i) for i in range(0, 9) if x[i] == 1]),
                                        axis=1)
        return Y_pred.loc[:, ['review_id', 'target']]


    def save(self, path: str) -> None:
        with open(os.path.join(path, 'pipeline.pickle'), 'wb') as file:
            pickle.dump(self.pipeline, file)
        with open(os.path.join(path, 'features.txt'), 'w') as file:
            for f in self.features:
                file.write(f + "\n")

    @staticmethod
    def load(path: str):
        obj = SimpleOVAPipeline()
        with open(os.path.join(path, 'pipeline.pickle'), 'rb') as file:
            obj.pipeline = pickle.load(file)
        with open(os.path.join(path, 'features.txt'), 'r') as file:
            obj.features = list(map(str.strip, file.readlines()))
        return obj

