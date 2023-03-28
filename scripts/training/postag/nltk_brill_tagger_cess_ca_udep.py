import json
from os import makedirs
from os.path import join, dirname
from random import shuffle

import nltk

from ovos_classifiers.tasks.tagger import OVOSBrillTagger, OVOSNgramTagger

TAGSET = "udep"
CORPUS = "cess_cat"
NAME = "brill"
TASK = "postag"

MODEL_META = {
    "corpus": "cess_cat_udep",
    "corpus_homepage": "https://github.com/OpenJarbas/biblioteca/blob/master/corpora/create_cess.py",
    "lang": "es",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": "Universal Dependencies",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["DefaultTagger", "AffixTagger", "UnigramTagger",
                        "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"],
    "train/test": "80/20"
}
TAGGER_META = {
    "corpus": "cess_cat_udep",
    "corpus_homepage": "https://github.com/OpenJarbas/biblioteca/blob/master/corpora/create_cess.py",
    "lang": "es",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-ngram-{TASK}",
    "tagset": "Universal Dependencies",
    "algo": "TrigramTagger",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"],
    "train/test": "80/20"
}

META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)
meta_path = join(META, MODEL_META["model_id"] + ".json")


def eagles_to_udep(tag):
    """EAGLES http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html"""
    tagdict = {
        'X': 'X',
        'Y': 'X',
        'i': 'X',
        'w': 'NOUN'  # time
    }
    if tag in tagdict:
        return tagdict[tag]
    tag = tag.lower().strip()
    if tag.startswith("v"):
        return 'VERB'
    if tag.startswith("p"):
        return 'PRON'
    if tag.startswith("n"):
        return 'NOUN'
    if tag.startswith("d"):
        return 'DET'
    if tag.startswith("a"):
        return 'ADJ'
    if tag.startswith("z"):
        return 'NUM'
    if tag.startswith("s"):
        return 'ADP'
    if tag.startswith("r"):
        return 'ADV'
    if tag.startswith("c"):
        return 'CONJ'
    if tag.startswith("f"):
        return '.'
    return tagdict.get(tag) or 'PART'


corpus = [
    [(w, eagles_to_udep(t)) for (w, t) in sent]
    for sent in nltk.corpus.cess_cat.tagged_sents()]

shuffle(corpus)

cutoff = int(len(corpus) * 0.9)
train_data = corpus[:cutoff]
test_data = corpus[cutoff:]

# train tagger
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
