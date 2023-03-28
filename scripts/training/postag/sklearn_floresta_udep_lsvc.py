import json
from os import makedirs
from os.path import join, dirname
from random import shuffle
from string import punctuation

import nltk
from sklearn.svm import LinearSVC

from ovos_classifiers.skovos.tagger import SklearnOVOSClassifierTagger

TAGSET = "udep"
CORPUS = "floresta"
NAME = "lsvc"
TASK = "postag"

MODEL_META = {
    "corpus": "floresta",
    "corpus_homepage": "http://www.linguateca.pt/Floresta",
    "tagset": "Universal Dependencies",
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


def convert_to_universal_tag(t, reverse=False):
    tagdict = {
        'n': "NOUN",
        'num': "NUM",
        'v-fin': "VERB",
        'v-inf': "VERB",
        'v-ger': "VERB",
        'v-pcp': "VERB",
        'pron-det': "PRON",
        'pron-indp': "PRON",
        'pron-pers': "PRON",
        'art': "DET",
        'adv': "ADV",
        'conj-s': "CONJ",
        'conj-c': "CONJ",
        'conj-p': "CONJ",
        'adj': "ADJ",
        'ec': "PRT",
        'pp': "ADP",
        'prp': "ADP",
        'prop': "NOUN",
        'pro-ks-rel': "PRON",
        'proadj': "PRON",
        'prep': "ADP",
        'nprop': "NOUN",
        'vaux': "VERB",
        'propess': "PRON",
        'v': "VERB",
        'vp': "VERB",
        'in': "X",
        'prp-': "ADP",
        'adv-ks': "ADV",
        'dad': "NUM",
        'prosub': "PRON",
        'tel': "NUM",
        'ap': "NUM",
        'est': "NOUN",
        'cur': "X",
        'pcp': "VERB",
        'pro-ks': "PRON",
        'hor': "NUM",
        'pden': "ADV",
        'dat': "NUM",
        'kc': "ADP",
        'ks': "ADP",
        'adv-ks-rel': "ADV",
        'npro': "NOUN",
    }
    if t in ["N|AP", "N|DAD", "N|DAT", "N|HOR", "N|TEL"]:
        t = "NUM"
    if reverse:
        if "|" in t: t = t.split("|")[0]
    else:
        if "+" in t: t = t.split("+")[1]
        if "|" in t: t = t.split("|")[1]
        if "#" in t: t = t.split("#")[0]
    t = t.lower()
    return tagdict.get(t, "." if all(tt in punctuation for tt in t) else t)


floresta = [[(w, convert_to_universal_tag(t)) for (w, t) in sent]
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
