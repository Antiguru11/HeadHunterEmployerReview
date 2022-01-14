import os

import pandas as pd
from scipy.stats import entropy

import utils


def make_features(source_df: pd.DataFrame,
              add_data_path: str = 'data') -> pd.DataFrame:
    # read add data
    cities_info_df = pd.read_csv(os.path.join(add_data_path, 'cities_info.csv'))

    # add city info (country, type, population and geo)
    source_df = source_df.merge(cities_info_df,
                                how='left',
                                left_on='city', right_on='name')

    # position 
    mask = source_df['position'].notna()
    source_df.loc[mask, 'position_preproc'] = (source_df[mask]['position']
                                               .apply(utils.preproc_str))

    # position lang
    en_mask = source_df['position_preproc'].str.contains('[a-zA-Z]+')
    ru_mask = source_df['position_preproc'].str.contains('[а-яА-Я]+')

    mask = source_df['position_preproc'].notna()
    source_df.loc[mask, 'position_lang'] = 'unknown'
    source_df.loc[mask & en_mask,'position_lang'] = 'en'
    source_df.loc[mask & ru_mask,'position_lang'] = 'ru'
    source_df.loc[mask & en_mask & ru_mask,'position_lang'] = 'en_ru'

    # position tokens count
    mask = source_df['position_preproc'].notna()
    source_df.loc[mask, 'position_tokens_count'] = (source_df[mask]['position_preproc']
                                                    .apply(lambda x: len(utils.tokenize(x))))

    # position have company name
    mask = source_df['position'].notna()
    source_df.loc[mask, 'position_have_company'] = (source_df[mask]['position']
                                                    .str.contains('[\*]+').astype(int))

    # rating
    rating_cols = ['salary_rating', 'team_rating', 'managment_rating',
                   'career_rating', 'workplace_rating', 'rest_recovery_rating', ]
    source_df['max_rating'] = source_df[rating_cols].max(axis=1)
    source_df['min_rating'] = source_df[rating_cols].min(axis=1)
    source_df['avg_rating'] = source_df[rating_cols].mean(axis=1)
    source_df['med_rating'] = source_df[rating_cols].median(axis=1)
    for val in range(1, 6):
        source_df[f'{str(val)}_rating_count'] = (source_df[rating_cols] == val).sum(axis=1)
    source_df['entropy_rating'] = entropy(source_df[[f'{str(v)}_rating_count' for v in range(1, 6)]] / 6, axis=1)

    return source_df

