import json
import pickle
import random
from os import makedirs
from os.path import join, dirname

import nltk
from nltk.corpus import brown
from nltk.tag.sequential import ClassifierBasedPOSTagger

TAGSET = "brown"
CORPUS = "brown"
NAME = "clf"
TASK = "postag"

MODEL_META = {
    "corpus": "brown",
    "lang": "en",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": "brown",
    "algo": "ClassifierBasedPOSTagger",
    "required_packages": ["nltk"]
}
META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

# initializing training and testing set
nltk.download('brown')

corpus = list(brown.tagged_sents())  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

tagger = ClassifierBasedPOSTagger(train=train_data)

a = tagger.evaluate(test_data)

MODEL_META["accuracy"] = a
print("Accuracy: ", a)
meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f, indent=4)

# save pickle
path = join(MODEL_BASE, MODEL_META["model_id"] + ".pkl")

with open(path, "wb") as f:
    pickle.dump(tagger, f)
