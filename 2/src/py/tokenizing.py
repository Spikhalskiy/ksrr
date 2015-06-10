from bs4 import BeautifulSoup
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import re
from spell_checking import SpellChecker

suffixes_stopwords = {"'ll", "'re", "'s", "wo", "do", "does", "ca", "were", "could", "is", "are"}

def _drop_html(html):
    return BeautifulSoup(html).get_text(separator=" ")


def _tokenize_for_nkd(s):
    if not isinstance(s, str) and not isinstance(s, unicode):
        s = str(s)

    s = _drop_html(s)
    return word_tokenize(s.lower())

word_pattern = re.compile(r"[a-z]+")

def _try_to_expand_complex_words(word):
    result = []
    parts = word_pattern.findall(word)
    if len(parts) == 1 and parts[0] == word:
        return result
    for part in parts:
        if len(part) > 2 and wn.synsets(part):
            result.append(part)
    return result

words_or_numbers = re.compile(r"[a-z0-9]{2}")
only_punctuation = re.compile(r"^[.,!?]+$")


def tokenize(s):
    stemmed = _tokenize_for_nkd(s)

    filtered_words = [w.lower() for w in stemmed if
                      w not in stopwords.words('english')
                      and w not in suffixes_stopwords
                      and (len(w) > 2 or len(w) == 2 and re.match(words_or_numbers, w))
                      and not re.match(only_punctuation, w)]

    add_words = []

    for word in filtered_words:
        add_words += _try_to_expand_complex_words(word)

    return filtered_words + add_words


def tokenize_with_spl(spl, porter=True, snowball=False):
    stemmer = nltk.stem.porter.PorterStemmer() if porter else None
    stemmer = nltk.stem.snowball.SnowballStemmer("english") if snowball else stemmer

    assert isinstance(spl, SpellChecker)

    def tokenizzzze(s):
        tokens = tokenize(s)
        reworked_result = []
        for token in tokens:
            token = spl.correct(token)
            if stemmer:
                if word_pattern.match(token):
                    token = stemmer.stem(token)
            reworked_result.append(token)
        return reworked_result

    return tokenizzzze


def tokenize_and_join(s):
    joined = " ".join(tokenize(s))
    joined = joined.strip()
    if len(joined) == 0:
        print "Tokenize problem for %s" % s
    return joined
