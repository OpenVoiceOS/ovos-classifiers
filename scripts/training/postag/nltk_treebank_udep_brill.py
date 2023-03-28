import json
import random
from os import makedirs
from os.path import join, dirname

import nltk
from nltk.corpus import treebank
from nltk.tag import brill

from ovos_classifiers.tasks.tagger import OVOSBrillTagger, OVOSNgramTagger

TAGSET = "udep"
CORPUS = "treebank"
NAME = "brill"
TASK = "postag"

MODEL_META = {
    "corpus": "treebank",
    "corpus_homepage": "https://www.nltk.org/nltk_data",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": "Universal Dependencies",
    "lang": "en",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["DefaultTagger", "RegexpTagger", "AffixTagger", "RegexpTagger",
                        "UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
TAGGER_META = {
    "corpus": "treebank",
    "corpus_homepage": "https://www.nltk.org/nltk_data",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-ngram-{TASK}",
    "tagset": "Universal Dependencies",
    "lang": "en",
    "algo": "TrigramTagger",
    "backoff_taggers": ["DefaultTagger", "RegexpTagger", "AffixTagger",
                        "RegexpTagger", "UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

# initializing training and testing set
nltk.download('treebank')
nltk.download('universal_tagset')

corpus = list(treebank.tagged_sents(tagset='universal'))  # 3914
random.shuffle(corpus)
train_data = corpus[:3000]
test_data = corpus[3000:]

# create tagger
patterns = [
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'(The|the|A|a|An|an)$', 'AT'),  # articles
    (r'.*able$', 'JJ'),  # adjectives
    (r'.*ness$', 'NN'),  # nouns formed from adjectives
    (r'.*ly$', 'RB'),  # adverbs
    (r'.*s$', 'NNS'),  # plural nouns
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*ed$', 'VBD'),  # past tense verbs
    (r'.*ment$', 'NN'),  # i.e. wonderment
    (r'.*ful$', 'JJ')  # i.e. wonderful
]
templates = [
    brill.Template(brill.Pos([-1])),
    brill.Template(brill.Pos([1])),
    brill.Template(brill.Pos([-2])),
    brill.Template(brill.Pos([2])),
    brill.Template(brill.Pos([-2, -1])),
    brill.Template(brill.Pos([1, 2])),
    brill.Template(brill.Pos([-3, -2, -1])),
    brill.Template(brill.Pos([1, 2, 3])),
    brill.Template(brill.Pos([-1]), brill.Pos([1])),
    brill.Template(brill.Word([-1])),
    brill.Template(brill.Word([1])),
    brill.Template(brill.Word([-2])),
    brill.Template(brill.Word([2])),
    brill.Template(brill.Word([-2, -1])),
    brill.Template(brill.Word([1, 2])),
    brill.Template(brill.Word([-3, -2, -1])),
    brill.Template(brill.Word([1, 2, 3])),
    brill.Template(brill.Word([-1]), brill.Word([1])),
]

# train tagger
clf = OVOSNgramTagger(default_tag="NN", regex_patterns=patterns)
clf.train(train_data)
clf.save(join(MODEL_BASE, TAGGER_META["model_id"] + ".pkl"))

acc = clf.score(test_data)

TAGGER_META["accuracy"] = acc
print("Accuracy of ngram tagger : ", TAGGER_META["accuracy"])  # 0.9224974329959557
meta_path = join(META, TAGGER_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(TAGGER_META, f, indent=4)

# train brill
clf = OVOSBrillTagger(default_tag="NN",
                      # templates=templates,
                      regex_patterns=patterns)
clf.train(train_data)
clf.save(join(MODEL_BASE, MODEL_META["model_id"] + ".pkl"))

acc = clf.score(test_data)
MODEL_META["accuracy"] = acc
print("Accuracy of Brill tagger : ", MODEL_META["accuracy"])  # 0.9353010205150772

meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f, indent=4)
