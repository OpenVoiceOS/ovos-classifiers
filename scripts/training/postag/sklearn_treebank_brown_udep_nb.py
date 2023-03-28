import json
import random
from os import makedirs
from os.path import join, dirname

import nltk
from nltk.corpus import brown, treebank
from sklearn.naive_bayes import MultinomialNB

from ovos_classifiers.tasks.tagger import OVOSClassifierTagger

TAGSET = "udep"
CORPUS = "brown+treebank"
NAME = "nb"
TASK = "postag"


MODEL_META = {
    "corpus": "brown+treebank",
    "model_id": f"sklearn-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": "Universal Dependencies",
    "corpus_homepage": "https://www.nltk.org/nltk_data",
    "lang": "en",
    "algo": "sklearn.naive_bayes.MultinomialNB",
    "required_packages": ["scikit-learn"]
}
META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

# initializing training and testing set
nltk.download('treebank')
nltk.download('brown')
nltk.download('universal_tagset')

corpus = list(treebank.tagged_sents(tagset='universal')) + \
         list(brown.tagged_sents(tagset='universal'))
random.shuffle(corpus)

cuttof = int(len(corpus) * 0.8)
train_data = corpus[:cuttof]
test_data = corpus[cuttof:]


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(tagged[index][0])
            y.append(tagged[index][1])

    return X, y


X, y = transform_to_dataset(train_data)
X = X[:10000]
y = y[:10000]

clf = OVOSClassifierTagger(MultinomialNB())
clf.train(X, y)

print('Training completed')

X_test, y_test = transform_to_dataset(test_data)
xx = X_test[-1]
yy = y_test[-1]
X_test = X_test[:10000]
y_test = y_test[:10000]

acc = clf.score(X_test, y_test)
MODEL_META["accuracy"] = acc
print("Accuracy:", acc)
# Accuracy:  0.9228

# save pickle
path = join(MODEL_BASE, MODEL_META["model_id"] + ".pkl")
clf.save(path)

meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f, indent=4)
