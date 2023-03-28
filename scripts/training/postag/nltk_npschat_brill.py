import json
import random
from os import makedirs
from os.path import join, dirname

import nltk
from nltk.corpus import nps_chat

from ovos_classifiers.tasks.tagger import OVOSBrillTagger, OVOSNgramTagger

TAGSET = "udep"
CORPUS = "nps_chat"
NAME = "brill"
TASK = "postag"

MODEL_META = {
    "corpus": "nps_chat",
    "corpus_homepage": "https://www.nltk.org/nltk_data",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": "Universal Dependencies",
    "lang": "en",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["DefaultTagger", "AffixTagger", "RegexpTagger",
                        "UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
TAGGER_META = {
    "corpus": "nps_chat",
    "corpus_homepage": "https://www.nltk.org/nltk_data",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-ngram-{TASK}",
    "tagset": "Universal Dependencies",
    "lang": "en",
    "algo": "TrigramTagger",
    "backoff_taggers": ["DefaultTagger", "AffixTagger",
                        "RegexpTagger", "UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

# initializing training and testing set
nltk.download('nps_chat')

corpus = list(nps_chat.tagged_posts(tagset="universal"))  # 3914
random.shuffle(corpus)

train_data = corpus[:3000]
test_data = corpus[3000:]

# train tagger
clf = OVOSNgramTagger(default_tag="NOUN")
clf.train(train_data)
clf.save(join(MODEL_BASE, TAGGER_META["model_id"] + ".pkl"))

acc = clf.score(test_data)

TAGGER_META["accuracy"] = acc
print("Accuracy of ngram tagger : ", TAGGER_META["accuracy"])  # 0.9224974329959557
meta_path = join(META, TAGGER_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(TAGGER_META, f, indent=4)

# train brill
clf = OVOSBrillTagger(default_tag="NOUN")
clf.train(train_data)
clf.save(join(MODEL_BASE, MODEL_META["model_id"] + ".pkl"))

acc = clf.score(test_data)
MODEL_META["accuracy"] = acc
print("Accuracy of Brill tagger : ", MODEL_META["accuracy"])  # 0.9353010205150772

meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f, indent=4)
