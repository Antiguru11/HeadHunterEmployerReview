import os
import sys
import yaml

import pandas as pd

from competition import featurize
from competition import pipeline


if __name__ == '__main__':
    name = sys.argv[1]
    print(name)

    config = yaml.load(open(f'configs/{name}.yaml', 'r'), yaml.Loader)

    features_path = f"data/{config['featurize']['name']}"
    artifacts_path = f"artifacts/{name}"     

    featurizer = featurize.functions[config['featurize']['name']]

    if os.path.exists(features_path):
        train_df = pd.read_csv(os.path.join(features_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(features_path, 'test.csv'))
    else:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')

        train_df = featurizer(train_df)
        test_df = featurizer(test_df)

        os.system('rm -fR {features_path} && mkdir {features_path}')
        train_df.to_csv(os.path.join(features_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(features_path, 'test.csv'), index=False)

    pipeline_type = pipeline.pipelines[config['pipeline']['name']]
    pipeline_obj = pipeline_type(verbose=True)
    results = pipeline_obj.fit(train_df, config['pipeline'])
    pipeline_obj.save(artifacts_path)
    print(results)

    preds = pipeline_obj.predict(test_df)
    preds.to_csv(f"submits/{name}.csv", index=False)
