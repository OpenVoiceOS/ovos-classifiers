import json
from os import makedirs
from os.path import join, dirname

from sklearn.tree import DecisionTreeClassifier

from ovos_classifiers.datasets import get_brown_trainset
from ovos_classifiers.skovos.tagger import SklearnOVOSClassifierTagger

TAGSET = "brown"
CORPUS = "brown"
NAME = "dtree"
TASK = "postag"

MODEL_META = {
    "corpus": "brown",
    "model_id": f"sklearn-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": "brown",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "algo": "sklearn.tree.DecisionTreeClassifier",
    "required_packages": ["scikit-learn"]
}
META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

# initializing training and testing set
(X, y), (X_test, y_test) = get_brown_trainset()

X = X[:10000]
y = y[:10000]
X_test = X_test[:10000]
y_test = y_test[:10000]

clf = SklearnOVOSClassifierTagger(DecisionTreeClassifier(criterion='entropy'))
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
