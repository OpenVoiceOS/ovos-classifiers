import json
from os import makedirs
from os.path import join, dirname

from sklearn.svm import LinearSVC

from ovos_classifiers.datasets import get_utterance_tags_trainset
from ovos_classifiers.tasks.classifier import OVOSClassifier

TAGSET = "utttags"
CORPUS = "utterance_tags_v0.1"
NAME = "lsvc"
PIPELINE = "questions_en"
TASK = "clf"

MODEL_META = {
    "corpus": CORPUS,
    "corpus_homepage": "https://github.com/OpenVoiceOS/ovos-datasets",
    "model_id": f"sklearn-{CORPUS}-{TAGSET}-{NAME}-{PIPELINE}-{TASK}",
    "tagset": CORPUS,
    "lang": "en",
    "algo": "sklearn.svm.LinearSVC",
    "required_packages": ["scikit-learn"]
}
META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "utttags")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

(X, y), (X_test, y_test) = get_utterance_tags_trainset()

clf = OVOSClassifier(PIPELINE, LinearSVC())
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
