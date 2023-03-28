import json
from os import makedirs, listdir
from os.path import join, dirname, isfile

from ovos_classifiers.datasets import get_utterance_tags_dataset
from ovos_classifiers.tasks.tagger import OVOSClassifierTagger

META = join(dirname(dirname(dirname(__file__))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(__file__))), "models", "utttags")
BENCHMARKS = join(dirname(dirname(dirname(__file__))), "models", "benchmarks", "utttags")
makedirs(META, exist_ok=True)
makedirs(BENCHMARKS, exist_ok=True)

model_accs = []

X, y = get_utterance_tags_dataset()

for m in listdir(META):
    with open(f"{META}/{m}") as f:
        data = json.load(f)
    if data.get("tagset", "") != "utterance_tags_v0.1":
        continue
    model = f"{MODEL_BASE}/{data['model_id']}.pkl"
    if not isfile(model):
        continue

    t = OVOSClassifierTagger.from_file(model)
    acc = t.score(X, y)

    model_accs.append((data['model_id'], data['accuracy'], acc))

readme = f"""

## Benchmark - Utterance Tags V0.1 corpus

dataset len: {len(X)}


| Model | Training Accuracy | Accuracy 	|
|-------|----------|----------|"""

model_accs = sorted(model_accs, key=lambda k: k[2], reverse=True)
for mid, tacc, acc in model_accs:
    readme += f"\n| {mid} | {tacc}  | {acc} |"

print(readme)

with open(f"{BENCHMARKS}/utttags.md", "w") as f:
    f.write(readme)
