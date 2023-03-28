import json
from os import makedirs
from os.path import dirname
from os.path import join
from random import shuffle

import nltk

from ovos_classifiers.tasks.tagger import OVOSBrillTagger, OVOSNgramTagger

TAGSET = "visl"
CORPUS = "floresta"
NAME = "brill"
TASK = "postag"


MODEL_META = {
    "corpus": "floresta",
    "corpus_homepage": "http://www.linguateca.pt/Floresta",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": "VISL (Portuguese)",
    "tagset_homepage": "https://visl.sdu.dk/visl/pt/symbolset-floresta.html",
    "lang": "pt",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}
TAGGER_META = {
    "corpus": "floresta",
    "corpus_homepage": "http://www.linguateca.pt/Floresta",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-ngram-{TASK}",
    "tagset": "VISL (Portuguese)",
    "tagset_homepage": "https://visl.sdu.dk/visl/pt/symbolset-floresta.html",
    "lang": "pt",
    "algo": "TrigramTagger",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
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

meta_path = join(META, MODEL_META["model_id"] + ".json")

clf = OVOSNgramTagger()
clf.train(train_data)
clf.save(join(MODEL_BASE, TAGGER_META["model_id"] + ".pkl"))

acc = clf.score(test_data)

TAGGER_META["accuracy"] = acc
print("Accuracy of ngram tagger : ", TAGGER_META["accuracy"])  # 0.9224974329959557

with open(meta_path, "w") as f:
    json.dump(TAGGER_META, f, indent=4)

# train brill
clf = OVOSBrillTagger()
clf.train(train_data)
clf.save(join(MODEL_BASE, MODEL_META["model_id"] + ".pkl"))

acc = clf.score(test_data)
MODEL_META["accuracy"] = acc
print("Accuracy of Brill tagger : ", MODEL_META["accuracy"])  # 0.9353010205150772

meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f, indent=4)
