import re
import enchant
import nltk
from tokenizing import tokenize_and_join
from nltk.corpus import wordnet as wn

import numpy as np
from sklearn.cross_validation import KFold

kf = KFold(6, n_folds=5)
for train, test in kf:
    print("%s %s" % (train, test))

X = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
print X[[1,2]]