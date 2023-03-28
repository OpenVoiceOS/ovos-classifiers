import json
import random
from os import makedirs
from os.path import join, dirname

import nltk
from nltk.corpus import brown

from ovos_classifiers.tasks.tagger import OVOSBrillTagger, OVOSNgramTagger

TAGSET = "brown"
CORPUS = "brown"
NAME = "brill"
TASK = "postag"

MODEL_META = {
    "corpus": "brown",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": "brown",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "algo": "nltk.brill.fntbl37",
    "backoff_taggers": ["DefaultTagger", "RegexpTagger", "AffixTagger", "RegexpTagger",
                        "UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}
TAGGER_META = {
    "corpus": "brown",
    "model_id": f"nltk-{CORPUS}-{TAGSET}-ngram-{TASK}",
    "tagset": "brown",
    "corpus_homepage": "http://www.hit.uib.no/icame/brown/bcm.html",
    "lang": "en",
    "algo": "TrigramTagger",
    "backoff_taggers": ["DefaultTagger", "RegexpTagger", "AffixTagger", "RegexpTagger",
                        "UnigramTagger", "BigramTagger", "TrigramTagger"],
    "required_packages": ["nltk"]
}

META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

# initializing training and testing set
nltk.download('brown')
nltk.download('universal_tagset')

corpus = [_ for _ in brown.tagged_sents()]  # 57340
random.shuffle(corpus)
cuttof = int(len(corpus) * 0.9)
train_data = corpus[:cuttof]
test_data = corpus[cuttof:]

# train tagger
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
clf = OVOSBrillTagger(default_tag="NN", regex_patterns=patterns)
clf.train(train_data)
clf.save(join(MODEL_BASE, MODEL_META["model_id"] + ".pkl"))

acc = clf.score(test_data)
MODEL_META["accuracy"] = acc
print("Accuracy of Brill tagger : ", MODEL_META["accuracy"])  # 0.9353010205150772

meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f, indent=4)
