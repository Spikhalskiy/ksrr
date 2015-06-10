import pandas as pd
from sklearn import cross_validation
from sklearn.svm import SVC, NuSVC
from features import LsaMapper
from resources import get_input, get_output
from spell_checking import SpellChecker
from tokenizing import tokenize
from utils import default_if_nan

train = pd.read_csv(get_input("train.csv"), encoding='utf-8')
test = pd.read_csv(get_input("test.csv"), encoding='utf-8')

# we dont need ID columns
idx = test.id.values.astype(int)
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# train = train.query("relevance_variance<1.1")

# create labels. drop useless columns
y = train.median_relevance.values

train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

all_dataset = pd.concat([train, test])

all_prod_data = list(all_dataset.apply(
    lambda x: '%s %s' % (default_if_nan(x['product_title']), default_if_nan(x['product_description'])),
    axis=1))

print "Fit spell checker"
spl = SpellChecker(tokenize, 4, 6)
spl.index_corpus_for_spell(all_prod_data)


featureMapper = LsaMapper(spl)

print "Fit features mapper"
featureMapper.fit(all_dataset)


print "Preparing train features"
train_features = featureMapper.transform(train)

print "Preparing test features"
test_features = featureMapper.transform(test)

print "Cross validation"
model = SVC(C=5)
scores = cross_validation.cross_val_score(model, train_features, y, cv=5)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

print "Fit model"
model = SVC(C=5)
model.fit(train_features, y)

print "Predict"
predictions = model.predict(test_features)

submission = pd.DataFrame({"id": idx, "prediction": predictions})
submission.to_csv(get_output("lsa.csv"), index=False)
