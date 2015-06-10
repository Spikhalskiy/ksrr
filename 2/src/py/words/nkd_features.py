from __future__ import division

import nltk
import numpy as np

from words import nkd
from tokenizing import tokenize


def __nkd_with_doc(query_stem, stemmed_doc, pairs_dict, terms_dict, docs_count=100000):
    min_nkd_for_query_stem = 2
    for doc_stem_index in xrange(len(stemmed_doc)):
        doc_stem = stemmed_doc[doc_stem_index]
        ndk_for_query_stem = nkd(query_stem, doc_stem, pairs_dict, terms_dict, docs_count)
        min_nkd_for_query_stem = min(ndk_for_query_stem, min_nkd_for_query_stem)
        if min_nkd_for_query_stem == 0:
            break
    return min_nkd_for_query_stem


def __minimize_nkd_with_doc(query_stem, stemmed_doc, pairs_dict, terms_dict, docs_count=100000):
    min_nkd_for_query_stem = __nkd_with_doc(query_stem, stemmed_doc, pairs_dict, terms_dict, docs_count)

    if min_nkd_for_query_stem >= 1:  # stem doesn't exist in corpus at all
        min_distance = 100500
        for doc_stem in stemmed_doc:
            distance = nltk.metrics.distance.edit_distance(query_stem, doc_stem)
            if distance < min_distance:
                min_distance = distance
        if (min_distance <= 1 and len(query_stem) >= 4) or min_distance < 0.1 * len(query_stem):
            min_nkd_for_query_stem = 0.05  # not 0 to make penalty
    return min_nkd_for_query_stem


def nkd_prepare_query_features(dataset, pairs_dict, terms_dict, all_dataset_size):
    nkd_features_for_queries = []
    for i, row in dataset.iterrows():
        query = row["normalized_query"]
        doc = row["normalized_description"]
        stemmed_query = tokenize(query)
        stemmed_doc = tokenize(doc)
        nkd_features_for_query = []
        for query_stem_index in xrange(len(stemmed_query)):
            query_stem = stemmed_query[query_stem_index]
            minimized_query_stem_ndk = __minimize_nkd_with_doc(query_stem, stemmed_doc, pairs_dict, terms_dict, all_dataset_size)
            nkd_features_for_query.append(minimized_query_stem_ndk)
        nkd_features_for_queries.append(nkd_features_for_query)
    return nkd_features_for_queries


def nkd_prepare_query_means(query_features):
    query_means = []
    for nkd_features_for_query in query_features:
        mean = sum(nkd_features_for_query) / len(nkd_features_for_query)
        query_means.append(mean)
    return np.reshape(np.asarray(query_means), (-1, 1))
