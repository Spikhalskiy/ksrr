"""
Beating the Benchmark
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""
from bs4 import BeautifulSoup
from numpy.core.multiarray import ndarray
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, NuSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from py.helpers import drop_html
import sklearn.metrics as metrics


def clean_html(line):
    line[1] = drop_html(str(line[1]))
    line[2] = drop_html(str(line[2]))


train = pd.read_csv('../input/train.csv')
cross = pd.read_csv('../input/cross.csv')
test = pd.read_csv('../input/test.csv')

# we dont need ID columns
idx = test.id.values.astype(int)
train = train.drop('id', axis=1)
cross = cross.drop('id', axis=1)
test = test.drop('id', axis=1)

# remove html
train.apply(clean_html, axis=1, raw=True)   # decreased results
cross.apply(clean_html, axis=1, raw=True)
test.apply(clean_html, axis=1, raw=True)

train = train.query("relevance_variance<1.1")

# create labels. drop useless columns
y = train.median_relevance.values
y_cross = cross.median_relevance.values
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

# do some lambda magic on text columns
train_data = list(train.apply(lambda x: '%s %s %s' % (x['query'], x['product_title'], x['product_description']), axis=1))
test_data = list(test.apply(lambda x: '%s %s %s' % (x['query'], x['product_title'], x['product_description']), axis=1))
cross_data = list(cross.apply(lambda x: '%s %s %s' % (x['query'], x['product_title'], x['product_description']), axis=1))

# the infamous tfidf vectorizer (Do you remember this one?)
tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), smooth_idf=1, sublinear_tf=1,
                      stop_words='english')

# Fit TFIDF
tfv.fit(train_data)
X = tfv.transform(train_data)
X_cross = tfv.transform(cross_data)
X_test = tfv.transform(test_data)

# LSA / SVD
svd = TruncatedSVD(n_components=200, n_iter=15)
X = svd.fit_transform(X)
X_cross = svd.transform(X_cross)
X_test = svd.transform(X_test)

# Scaling the data is important prior to SVM
scl = StandardScaler()
X = scl.fit_transform(X)
X_cross = scl.transform(X_cross)
X_test = scl.transform(X_test)

model = SVC(C=14)


# Fit SVM Model
model.fit(X, y)

print "accuracy: %s" % metrics.accuracy_score(y_cross, model.predict(X_cross))

preds = model.predict(X_test)

# Create your first submission file
submission = pd.DataFrame({"id": idx, "prediction": preds})
submission.to_csv("beating_the_benchmark_yet_again.csv", index=False)

