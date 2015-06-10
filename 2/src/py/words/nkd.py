from collections import defaultdict
import math

import pandas as pd

from tokenizing import tokenize


def key(key1, key2):
    if key1 > key2:
        return key1 + "_X_" + key2
    else:
        return key2 + "_X_" + key1

def index_document(s, pairs_dict, terms_dict):
    # Creates half the matrix of pairwise tokens
    # This fits into memory, else we have to choose a Count-min Sketch probabilistic counter
    doc_terms = set()
    doc_pairs = set()

    tokens = tokenize(s)
    for x in xrange(len(tokens)):
        doc_terms.add(tokens[x])
        for y in xrange(x + 1, len(tokens)):
            doc_pairs.add(key(tokens[x], tokens[y]))

    for term in doc_terms:
        terms_dict[term] += 1
    for terms_pair in doc_pairs:
        pairs_dict[terms_pair] += 1


def index_corpus_ds(ds, column_name):
    assert isinstance(ds, pd.DataFrame)
    # Create our count dictionary and fill it with train and test set (pairwise) token counts
    pairs_dict = defaultdict(int)
    terms_dict = defaultdict(int)
    for i, row in ds.iterrows():
        s = row[column_name]
        index_document(s, pairs_dict, terms_dict)
    return pairs_dict, terms_dict


def nkd(token_x, token_y, pairs_dict, terms_dict, docs_count=100000):
    # Returns the semantic Normalized Kaggle Distance between two tokens
    if terms_dict[token_x] == 0 or terms_dict[token_y] == 0 or pairs_dict[key(token_x, token_y)] == 0:
        return 1
    else:
        logcount_x = math.log(terms_dict[token_x], 2)
        logcount_y = math.log(terms_dict[token_y], 2)
        logcount_xy = math.log(pairs_dict[key(token_x, token_y)], 2)
        log_index_size = math.log(docs_count, 2)  # fixed guesstimate
        nkd_distance = (max(logcount_x, logcount_y) - logcount_xy) / (log_index_size - min(logcount_x, logcount_y))
        # nkd = (max(logcount_x, logcount_y) - logcount_xy) / max(logcount_x, logcount_y)

        if nkd_distance > 1:
            nkd_distance = 1

        # assert 1 >= nkd >= 0
        return nkd_distance


