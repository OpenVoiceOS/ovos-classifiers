import json
from os import makedirs
from os.path import dirname, join
from random import shuffle

import nltk

from ovos_classifiers.tasks.tagger import OVOSBrillTagger, OVOSNgramTagger

TAGSET = "visl" # ?
CORPUS = "mcmorpho"
NAME = "brill"
TASK = "postag"

MODEL_META = {
    "corpus": "macmorpho",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/macmorpho/",
    "tagset": "",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf",
    "lang": "pt",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["DefaultTagger", "RegexpTagger", "AffixTagger", "RegexpTagger",
                        "UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
TAGGER_META = {
    "corpus": "macmorpho",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-ngram-{TASK}",
    "corpus_homepage": "http://www.nilc.icmc.usp.br/macmorpho/",
    "tagset": "",
    "tagset_homepage": "http://www.nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf",
    "lang": "pt",
    "algo": "TrigramTagger",
    "backoff_taggers": ["DefaultTagger", "RegexpTagger", "AffixTagger", "RegexpTagger",
                        "UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}

META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

meta_path = join(META, MODEL_META["model_id"] + ".json")

nltk.download('mac_morpho')


def clean_tag(t, ):
    if "|" in t:
        t = t.split("|")[0]
    return t


dataset = [[(w, clean_tag(t)) for (w, t) in sent]
           for sent in nltk.corpus.mac_morpho.tagged_sents()]

shuffle(dataset)
print(dataset[0])

cutoff = int(len(dataset) * 0.9)
train_data = dataset[:cutoff]
test_data = dataset[cutoff:]

patterns = [
    (r"^[nN][ao]s?$", "ADP"),
    (r"^[dD][ao]s?$", "ADP"),
    (r"^[pP]el[ao]s?$", "ADP"),
    (r"^[nN]est[ae]s?$", "ADP"),
    (r"^[nN]um$", "ADP"),
    (r"^[nN]ess[ae]s?$", "ADP"),
    (r"^[nN]aquel[ae]s?$", "ADP"),
    (r"^\xe0$", "ADP"),
]

clf = OVOSNgramTagger(default_tag="N", regex_patterns=patterns)
clf.train(train_data)
clf.save(join(MODEL_BASE, TAGGER_META["model_id"] + ".pkl"))

acc = clf.score(test_data)

TAGGER_META["accuracy"] = acc
print("Accuracy of ngram tagger : ", TAGGER_META["accuracy"])  # 0.9224974329959557

with open(meta_path, "w") as f:
    json.dump(TAGGER_META, f, indent=4)

# train brill
clf = OVOSBrillTagger(default_tag="N", regex_patterns=patterns)
clf.train(train_data)
clf.save(join(MODEL_BASE, MODEL_META["model_id"] + ".pkl"))

acc = clf.score(test_data)
MODEL_META["accuracy"] = acc
print("Accuracy of Brill tagger : ", MODEL_META["accuracy"])  # 0.9353010205150772

meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f, indent=4)
