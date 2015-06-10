import pandas as pd
import numpy as np

from tokenizing import tokenize

__author__ = 'Dmitry Spikhalskiy <dmitry@spikhalskiy.com>'

def prepare_all_words_in_title_feature(dataset, query_column_name, title_column_name):
    assert isinstance(dataset, pd.DataFrame)

    is_all_words = []
    for i, row in dataset.iterrows():
        query_stems = tokenize(row[query_column_name])
        title_stems = tokenize(row[title_column_name])
        found_different_term = False
        for query_term in query_stems:
            if query_term not in title_stems:
                found_different_term = True
                break
        is_all_words.append(0 if found_different_term else 1)
    return np.reshape(np.asarray(is_all_words), (-1, 1))
