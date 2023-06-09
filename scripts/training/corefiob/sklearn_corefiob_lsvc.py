import json
import random
from os import makedirs
from os.path import join, dirname

from ovos_classifiers.datasets import get_coref_iob_trainset
from sklearn.svm import LinearSVC

from ovos_classifiers.skovos.tagger import SklearnOVOSClassifierTagger

TAGSET = "corefiob"
CORPUS = "corefiob_v0.1"
NAME = "lsvc"
TASK = "iob"

MODEL_META = {
    "corpus": CORPUS,
    "model_id": f"sklearn-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": TAGSET,
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "algo": "sklearn.svm.LinearSVC",
    "required_packages": ["scikit-learn"]
}
META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

# initializing training and testing set

(X, y), (X_test, y_test) = get_coref_iob_trainset()



clf = SklearnOVOSClassifierTagger(LinearSVC(), pipeline_id="pronouns_en")
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
