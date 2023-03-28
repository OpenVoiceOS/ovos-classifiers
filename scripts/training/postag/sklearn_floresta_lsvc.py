import json
from os import makedirs
from os.path import join, dirname
from random import shuffle

import nltk
from sklearn.svm import LinearSVC

from ovos_classifiers.skovos.tagger import SklearnOVOSClassifierTagger

TAGSET = "visl"
CORPUS = "floresta"
NAME = "lsvc"
TASK = "postag"

MODEL_META = {
    "corpus": "floresta",
    "corpus_homepage": "http://www.linguateca.pt/Floresta",
    "tagset": "VISL (Portuguese)",
    "tagset_homepage": "https://visl.sdu.dk/visl/pt/symbolset-floresta.html",
    "lang": "pt",
    "model_id": f"sklearn-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "algo": "sklearn.svm.LinearSVC",
    "required_packages": ["scikit-learn"]
}
META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

nltk.download('floresta')


def clean_tag(t):
    if "+" in t: t = t.split("+")[1]
    if "|" in t: t = t.split("|")[1]
    if "#" in t: t = t.split("#")[0]
    t = t.lower()
    return t


floresta = [[(w, clean_tag(t)) for (w, t) in sent]
            for sent in nltk.corpus.floresta.tagged_sents()]
shuffle(floresta)

cutoff = int(len(floresta) * 0.9)
train_data = floresta[:cutoff]
test_data = floresta[cutoff:]


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

clf = SklearnOVOSClassifierTagger(LinearSVC())
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
