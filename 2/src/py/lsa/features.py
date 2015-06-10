import nltk
import re
from spell_checking import SpellChecker
from numpy.core.multiarray import ndarray
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, ProjectedGradientNMF
from sklearn.preprocessing import StandardScaler

from tokenizing import tokenize, tokenize_with_spl
from utils import default_if_nan

word_pattern = re.compile(r"[a-z]+")

class LsaMapper:
    def __init__(self, spl):
        self.spl = spl
        self.tfv = TfidfVectorizer(min_df=3, max_features=None,
                                   tokenizer=tokenize_with_spl(spl, porter=True),
                                   strip_accents='unicode', analyzer='word',
                                   ngram_range=(1, 3), smooth_idf=1, sublinear_tf=1)
        self.svd = TruncatedSVD(n_components=300)
        # svd = ProjectedGradientNMF(n_components=240, random_state=True, sparseness="components")

        self.scl = StandardScaler()

    def fit(self, dataset, y=None):
        prod_data = list(dataset.apply(
            lambda x: '%s %s %s' % (default_if_nan(x['product_title']), default_if_nan(x['product_title']),
                                    default_if_nan(x['product_description'])),
            axis=1))

        print "         Prepare tf-idf"
        X = self.tfv.fit_transform(prod_data)

        print "         Prepare SVD"
        X = self.svd.fit_transform(X)

        print "         Prepare Scaller"
        self.scl.fit(X)
        return self

    def transform(self, dataset):
        assert isinstance(dataset, pd.DataFrame)
        prod_data = list(dataset.apply(
            lambda x: '%s %s %s' % (default_if_nan(x['product_title']), default_if_nan(x['product_title']),
                                    default_if_nan(x['product_description'])),
            axis=1))
        query_data = list(dataset.apply(lambda x: str(x['query']), axis=1))

        X = self.tfv.transform(prod_data)
        Q = self.tfv.transform(query_data)

        X = self.svd.transform(X)
        Q = self.svd.transform(Q)

        X = self.scl.transform(X)
        Q = self.scl.transform(Q)

        return prepare_features(X, Q)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def prepare_features(X, Q):
    assert isinstance(X, ndarray)
    assert isinstance(Q, ndarray)
    # dist = DistanceMetric.get_metric('euclidean')
    # assert isinstance(dist, DistanceMetric)
    # pairwise_distance = dist.pairwise(X, Q)
    # assert isinstance(pairwise_distance, ndarray)
    # diagonal = pairwise_distance.diagonal()
    # assert isinstance(diagonal, ndarray)
    # diagonal = np.reshape(diagonal, (-1, 1))
    return np.concatenate([X, Q
                           # , X - Q, diagonal
                           ], axis=1)
