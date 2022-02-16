import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import entropy

from .utils import preproc_str, tokenize, ru_normalize


def _text_featurize(source_df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    mask = source_df[column_name].notna()
    source_df.loc[mask, f'{column_name}_preproc'] = (source_df[mask][column_name]
                                                     .apply(preproc_str))

    # lang
    en_mask = source_df[f'{column_name}_preproc'].str.contains('[a-zA-Z]+')
    ru_mask = source_df[f'{column_name}_preproc'].str.contains('[а-яА-Я]+')

    mask = source_df[f'{column_name}_preproc'].notna()
    source_df.loc[mask, f'{column_name}_lang'] = 'unknown'
    source_df.loc[mask & en_mask,f'{column_name}_lang'] = 'en'
    source_df.loc[mask & ru_mask,f'{column_name}_lang'] = 'ru'
    source_df.loc[mask & en_mask & ru_mask,f'{column_name}_lang'] = 'en_ru'

    # tokens count
    mask = source_df[f'{column_name}_preproc'].notna()
    source_df.loc[mask, f'{column_name}_tokens_count'] = (source_df[mask][f'{column_name}_preproc']
                                                          .apply(lambda x: len(tokenize(x))))

    # have company name flag
    mask = source_df[column_name].notna()
    source_df.loc[mask, f'{column_name}_have_company'] = (source_df[mask][column_name]
                                                          .str.contains('[\*]+')
                                                          .astype(int))

    # empty flag
    source_df[f'{column_name}_empty'] = source_df[column_name].isna().astype(int)

    # unique token count
    mask = source_df[f'{column_name}_preproc'].notna()
    source_df.loc[mask, f'{column_name}_unique_tokens_count'] = (source_df[mask][f'{column_name}_preproc']
                                                                 .apply(lambda x: len(set([t
                                                                                           for t in tokenize(x)
                                                                                           if re.search('[а-яА-Я]+', t)]))))

    # no alpha count
    mask = source_df[column_name].notna()
    source_df.loc[mask, f'{column_name}_no_alpha_count'] = (source_df[mask][column_name]
                                                            .str.strip(' *')
                                                            .apply(lambda x: len(x) - sum(map(str.isalpha, x))))

    return source_df


def make_base_features(source_df: pd.DataFrame) -> pd.DataFrame:
    # read add data
    cities_info_df = pd.read_csv(os.path.join('data', 'cities_info.csv'))

    # add city info (country, type, population and geo)
    source_df = source_df.merge(cities_info_df,
                                how='left',
                                left_on='city', right_on='name')

    # position 
    source_df = _text_featurize(source_df, 'position')

    # positive 
    source_df = _text_featurize(source_df, 'positive')

    # negative 
    source_df = _text_featurize(source_df, 'negative')

    # rating
    rating_cols = ['salary_rating', 'team_rating', 'managment_rating',
                   'career_rating', 'workplace_rating', 'rest_recovery_rating', ]
    source_df['max_rating'] = source_df[rating_cols].max(axis=1)
    source_df['min_rating'] = source_df[rating_cols].min(axis=1)
    source_df['avg_rating'] = source_df[rating_cols].mean(axis=1)
    source_df['med_rating'] = source_df[rating_cols].median(axis=1)
    for val in range(1, 6):
        source_df[f'{str(val)}_rating_count'] = (source_df[rating_cols] == val).sum(axis=1)
        source_df[f'{str(val)}_rating_frac'] = source_df[f'{str(val)}_rating_count'] / 6
    source_df['entropy_rating'] = entropy(source_df[[f'{str(v)}_rating_count' for v in range(1, 6)]] / 6, axis=1)

    # other
    source_df['positive_tokens_frac'] = (source_df['positive_tokens_count'] 
                                         / source_df['negative_tokens_count'].replace(0, np.nan))
    
    source_df['positive_unique_tokens_frac'] = (source_df['positive_unique_tokens_count'] 
                                                / source_df['positive_tokens_count'].replace(0, np.nan))
    source_df['negative_unique_tokens_frac'] = (source_df['negative_unique_tokens_count'] 
                                                / source_df['negative_tokens_count'].replace(0, np.nan))

    source_df['positive_no_alpha_frac'] = (source_df['positive_no_alpha_count'] 
                                           / source_df['positive'].str.strip(' *').str.len().replace(0, np.nan))
    source_df['negative_no_alpha_frac'] = (source_df['negative_no_alpha_count'] 
                                           / source_df['negative'].str.strip(' *').str.len().replace(0, np.nan))

    source_df.loc[:, 'positive_negative'] = (source_df['positive'].fillna('') 
                                             + ' ' +  source_df['negative'].fillna(''))

    source_df.loc[:, 'have_unknown_symbol'] = (source_df['positive_negative']
                                               .str.contains('\xa0')
                                               .astype(int))

    return source_df


functions = {'base': make_base_features, }


if __name__ == '__main__':
    data_path, filename, function, = sys.argv[1:]

    source_df = pd.read_csv(os.path.join(data_path, filename + '.csv'))
    features_df = functions[function](source_df, data_path)

    features_df.to_csv(os.path.join(data_path,
                                    filename + f'_{function}_features.csv'),
                       index=False)