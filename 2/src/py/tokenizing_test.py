import codecs
import nltk
import pandas as pd
import re
from resources import get_input, get_output
from spell_checking import SpellChecker
from tokenizing import tokenize
from utils import default_if_nan

__author__ = 'Dmitry Spikhalskiy <dmitry@spikhalskiy.com>'

word_pattern = re.compile(r"[a-z]+")

def tokenize_with_spl(spl):
    stemmer = nltk.stem.porter.PorterStemmer()
    assert isinstance(spl, SpellChecker)

    def tokenizzzze(s):
        tokens = tokenize(s)
        reworked_result = []
        for token in tokens:
            token = spl.correct(token)
            if word_pattern.match(token):
                token = stemmer.stem_word(token)
            reworked_result.append(token)
        return reworked_result

    return tokenizzzze

train = pd.read_csv(get_input("train.csv"), encoding='utf-8')
assert isinstance(train, pd.DataFrame)

train = train.drop('id', axis=1)
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

all_prod_data = list(train.apply(
    lambda x: '%s %s' % (default_if_nan(x['product_title']), default_if_nan(x['product_description'])),
    axis=1))

spl = SpellChecker(tokenize, 5, 7)
spl.index_corpus_for_spell(all_prod_data)

tokenizinch = tokenize_with_spl(spl)

with codecs.open(get_output("tokenized.csv"), "w", "utf-8") as f:
    for description in all_prod_data:
        print >>f, " ".join(tokenizinch(description))
