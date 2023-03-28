import json
from os import makedirs
from os.path import dirname, join
from random import shuffle
from string import punctuation

import nltk

from ovos_classifiers.tasks.tagger import OVOSBrillTagger, OVOSNgramTagger

TAGSET = "udep"
CORPUS = "floresta+mcmorpho"
NAME = "brill"
TASK = "postag"

MODEL_META = {
    "corpus": "floresta + macmorpho",
    "lang": "pt",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "corpus_homepage": "https://www.nltk.org/nltk_data",
    "tagset": "Universal Dependencies",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["DefaultTagger", "RegexpTagger", "AffixTagger", "RegexpTagger",
                        "UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
TAGGER_META = {
    "corpus": "floresta + macmorpho",
    "lang": "pt",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-ngram-{TASK}",
    "corpus_homepage": "https://www.nltk.org/nltk_data",
    "tagset": "Universal Dependencies",
    "algo": "nltk.brill.fntbl37",
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
nltk.download('floresta')

path = join(MODEL_BASE, MODEL_META["model_id"] + ".pkl")


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


mac_morpho = [
    [(w, convert_to_universal_tag(t, reverse=True)) for (w, t) in sent]
    for sent in nltk.corpus.mac_morpho.tagged_sents()]

floresta = [[(w, convert_to_universal_tag(t)) for (w, t) in sent]
            for sent in nltk.corpus.floresta.tagged_sents()]

dataset = floresta + mac_morpho
shuffle(dataset)

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

clf = OVOSNgramTagger(default_tag="NOUN", regex_patterns=patterns)
clf.train(train_data)
clf.save(join(MODEL_BASE, TAGGER_META["model_id"] + ".pkl"))

acc = clf.score(test_data)

TAGGER_META["accuracy"] = acc
print("Accuracy of ngram tagger : ", TAGGER_META["accuracy"])  # 0.9224974329959557

with open(meta_path, "w") as f:
    json.dump(TAGGER_META, f, indent=4)

# train brill
clf = OVOSBrillTagger(default_tag="NOUN", regex_patterns=patterns)
clf.train(train_data)
clf.save(join(MODEL_BASE, MODEL_META["model_id"] + ".pkl"))

acc = clf.score(test_data)
MODEL_META["accuracy"] = acc
print("Accuracy of Brill tagger : ", MODEL_META["accuracy"])  # 0.9353010205150772

meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f, indent=4)
