from __future__ import division
import time

import pandas as pd
from sklearn import metrics, cross_validation
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from nkd import index_corpus_ds
from nkd_features import nkd_prepare_query_features, nkd_prepare_query_means
from naive_features import prepare_all_words_in_title_feature
from tokenizing import tokenize_and_join

__author__ = 'Dmitry Spikhalskiy <dmitry@spikhalskiy.com>'


def prepare_additional_columns(dataset):
    for i, row in dataset.iterrows():
        normalized_product_title = tokenize_and_join(str(row['product_title']))
        dataset.set_value(i, "normalized_product_title", normalized_product_title)
        normalized_product_description = tokenize_and_join(str(row['product_description']))
        dataset.set_value(i, "normalized_product_description", normalized_product_description)

        dataset.set_value(i, "normalized_description",
                          '%s %s' % (
                              normalized_product_title, normalized_product_description))
        dataset.set_value(i, "normalized_query", tokenize_and_join(str(row['query'])))

def make_classifier():
    # model = SVC(C=14)
    model = DecisionTreeClassifier(min_samples_leaf=200, max_depth=4)
    return model

if __name__ == "__main__":
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    # we dont need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)

    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

    print "Normalize train dataset..."
    start = time.time()
    prepare_additional_columns(train)
    end = time.time()
    print "Normalize train dataset took %s seconds" % (end - start)

    print "Normalize test dataset..."
    start = time.time()
    prepare_additional_columns(test)
    end = time.time()
    print "Normalize test dataset took %s seconds" % (end - start)

    all_dataset = pd.concat([train, test])

    print "[All query words in title]: prepare train features"
    start = time.time()
    all_terms_in_title = prepare_all_words_in_title_feature(train, "normalized_query", "normalized_product_title")
    end = time.time()
    print "[All query words in title]: prepare train features took %s seconds" % (end - start)

    print "[NKD] Preparing corpuses from products..."
    start = time.time()
    pairs_dict, terms_dict = index_corpus_ds(all_dataset, "normalized_description")
    end = time.time()
    print "[NKD] Preparing distances from products took %s seconds" % (end - start)

    print "[NKD] Preparing metrics for train queries"
    start = time.time()
    nkd_features_for_queries = nkd_prepare_query_features(train, pairs_dict, terms_dict, len(all_dataset))
    nkd_query_means = nkd_prepare_query_means(nkd_features_for_queries)
    end = time.time()
    print "[NKD] Preparing distances for train queries took %s seconds" % (end - start)

    print "Preparing all train features"
    start = time.time()
    train_features = np.concatenate([all_terms_in_title, nkd_query_means], axis=1)
    np.savetxt("train_features_dump.txt", train_features)
    end = time.time()
    print "Preparing all train features took %s seconds" % (end - start)

    print "Cross validation"
    start = time.time()

    model = make_classifier()

    scores = cross_validation.cross_val_score(model, train_features, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    end = time.time()
    print "Cross validation took %s seconds" % (end - start)

    print "Learn model"
    start = time.time()

    model = make_classifier()
    model.fit(train_features, y)

    dotfile = open("dtree2.dot", 'w')
    tree.export_graphviz(model.tree_, out_file=dotfile, feature_names=["all_terms_in_title", "distance"])
    dotfile.close()

    predict = model.predict(train_features)
    print "accuracy: %s" % metrics.accuracy_score(y, predict)

    end = time.time()
    print "Learn model took %s seconds" % (end - start)

    print "[All query words in title]: prepare test features"
    start = time.time()
    all_terms_in_title = prepare_all_words_in_title_feature(test, "normalized_query", "normalized_product_title")
    end = time.time()
    print "[All query words in title]: prepare test features took %s seconds" % (end - start)

    print "[NKD] Preparing distances for test queries"
    start = time.time()
    nkd_features_for_queries = nkd_prepare_query_features(test, pairs_dict, terms_dict, len(all_dataset))
    nkd_query_means = nkd_prepare_query_means(nkd_features_for_queries)
    end = time.time()
    print "[NKD] Preparing distances for test queries took %s seconds" % (end - start)

    print "Preparing all train features"
    start = time.time()
    test_features = np.concatenate([all_terms_in_title, nkd_query_means], axis=1)
    end = time.time()
    print "Preparing all train features took %s seconds" % (end - start)

    print "Predict test set"
    start = time.time()
    prediction_result = model.predict(test_features)
    end = time.time()
    print "Predict test set took %s seconds" % (end - start)

    submission = pd.DataFrame({"id": idx, "prediction": prediction_result})
    submission.to_csv("beating_the_benchmark_yet_again.csv", index=False)

