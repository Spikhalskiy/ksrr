from collections import defaultdict
import re
import nltk

__author__ = 'Dmitry Spikhalskiy <dmitry@spikhalskiy.com>'

def is_levenshtein_similar(s1, s2, max_distance):
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s1) - len(s2) > max_distance:
        return False
    if len(set(s1) - set(s2)) > max_distance:
        return False
    return nltk.metrics.distance.edit_distance(s1, s2) <= max_distance



class SpellChecker:
    word_pattern = re.compile(r'^[a-z]+$')

    terms_dict = defaultdict(int)

    # cached results
    right_spell = set()
    spell_check_result = dict()

    def __init__(self, tokenizer, len_min_limit, len_min_limit_2):
        self.tokenizer = tokenizer
        self.len_min_limit = len_min_limit
        self.len_min_limit_2 = len_min_limit_2

    def index_corpus_for_spell(self, corpus):
        for sentence in corpus:
            tokens = self.tokenizer(sentence)
            for token in tokens:
                if len(token) >= self.len_min_limit and self.word_pattern.match(token):
                    self.terms_dict[token] += 1

    def correct(self, word):
        word = word.lower()
        if len(word) < self.len_min_limit:
            return word

        full_word = self.word_pattern.match(word)
        if not full_word:
            if len(word) >= self.len_min_limit + 1 and self.word_pattern.match(word[1:-1]):
                if self.word_pattern.match(word[1:]):
                    word = word[1:]
                elif self.word_pattern.match(word[:-1]):
                    word = word[:-1]
                else:
                    word = word[1:-1]
            else:
                return word

        if word in self.right_spell:
            return word

        if word in self.spell_check_result:
            return self.spell_check_result[word]

        word_docs_count = self.terms_dict[word]

        max_distance = 1 if len(word) < self.len_min_limit_2 else 2
        current_best_word = word
        current_best_count = word_docs_count

        for another_word in self.terms_dict:
            another_word_docs_count = self.terms_dict[another_word]
            if another_word_docs_count > 1.8 * word_docs_count \
                    and another_word_docs_count > current_best_count \
                    and is_levenshtein_similar(word, another_word, max_distance):
                current_best_word = another_word
                current_best_count = another_word_docs_count

        if current_best_word == word:
            self.right_spell.add(word)
        else:
            self.spell_check_result[word] = current_best_word

        return current_best_word
