import json
import random
from os import makedirs, listdir
from os.path import join, dirname, isfile

import nltk
from nltk.corpus import treebank

from ovos_classifiers.tasks.tagger import OVOSBrillTagger, OVOSNgramTagger, OVOSClassifierTagger

META = join(dirname(dirname(dirname(__file__))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(__file__))), "models", "postag")
BENCHMARKS = join(dirname(dirname(dirname(__file__))), "models", "benchmarks", "postag")
makedirs(META, exist_ok=True)
makedirs(BENCHMARKS, exist_ok=True)

# initializing training and testing set
nltk.download('treebank')

corpus = list(treebank.tagged_sents())  # 3914
random.shuffle(corpus)


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(tagged[index][0])
            y.append(tagged[index][1])

    return X, y



model_accs = []

for m in listdir(META):
    with open(f"{META}/{m}") as f:
        data = json.load(f)
    if data.get("tagset", "") != "Penn Treebank":
        continue
    model = f"{MODEL_BASE}/{data['model_id']}.pkl"

    if not isfile(model):
        continue
    if data["algo"] == "TrigramTagger":
        t = OVOSNgramTagger.from_file(model)
        acc = t.score(corpus)
    elif data["algo"] == "nltk.brill.fntbl37":
        t = OVOSBrillTagger.from_file(model)
        acc = t.score(corpus)
    else:
        t = OVOSClassifierTagger.from_file(model)
        X_test, y_test = transform_to_dataset(corpus)

        # HACK - avoid run out of memory
        X = X_test[:10000]
        y = y_test[:10000]
        # /end HACK

        acc = t.score(X, y)

    model_accs.append((data['model_id'], data['accuracy'], acc))

readme = f"""

## Benchmark - nltk treebank corpus

dataset len: {len(corpus)}


| Model | Training Accuracy | Accuracy 	|
|-------|----------|----------|"""

model_accs = sorted(model_accs, key=lambda k: k[2], reverse=True)
for mid, tacc, acc in model_accs:
    readme += f"\n| {mid} | {tacc}  | {acc} |"

print(readme)

with open(f"{BENCHMARKS}/treebank.md", "w") as f:
    f.write(readme)
