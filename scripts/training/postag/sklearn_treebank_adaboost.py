import json
from os import makedirs
from os.path import join, dirname

from sklearn.ensemble import AdaBoostClassifier

from ovos_classifiers.datasets import get_treebank_trainset
from ovos_classifiers.tasks.tagger import OVOSClassifierTagger

TAGSET = "pentree"
CORPUS = "treebank"
NAME = "ada"
TASK = "postag"

MODEL_META = {
    "corpus": "treebank",
    "corpus_homepage": "https://www.nltk.org/nltk_data",
    "model_id": f"sklearn-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": "Penn Treebank",
    "lang": "en",
    "algo": "sklearn.ensemble.AdaBoostClassifier",
    "required_packages": ["scikit-learn"]
}
META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

# initializing training and testing set
(X, y), (X_test, y_test) = get_treebank_trainset()

X = X[:10000]
y = y[:10000]
X_test = X_test[:10000]
y_test = y_test[:10000]

clf = OVOSClassifierTagger(AdaBoostClassifier(n_estimators=50))
clf.train(X, y)

print('Training completed')

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
