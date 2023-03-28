import json
from os import makedirs
from os.path import join, dirname
from random import shuffle

import nltk

from ovos_classifiers.tasks.tagger import OVOSBrillTagger, OVOSNgramTagger

TAGSET = "eagles"
CORPUS = "cess_cat"
NAME = "brill"
TASK = "postag"

MODEL_META = {
    "corpus": "cess_cat",
    "corpus_homepage": "https://web.archive.org/web/20121023154634/http://clic.ub.edu/cessece/",
    "lang": "ca",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": "EAGLES",
    "tagset_homepage": "http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["AffixTagger", "UnigramTagger",
                        "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
TAGGER_META = {
    "corpus": "cess_cat",
    "corpus_homepage": "https://web.archive.org/web/20121023154634/http://clic.ub.edu/cessece/",
    "lang": "ca",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-ngram-{TASK}",
    "tagset": "EAGLES",
    "tagset_homepage": "http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html",
    "algo": "TrigramTagger",
    "backoff_taggers": ["AffixTagger", "UnigramTagger", "BigramTagger",
                        "TrigramTagger"],
    "required_packages": ["nltk"]
}

META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

# EAGLES
# http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html
nltk.download('cess_cat')

corpus = [sent for sent in nltk.corpus.cess_cat.tagged_sents()]
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
meta_path = join(META, TAGGER_META["model_id"] + ".json")
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
