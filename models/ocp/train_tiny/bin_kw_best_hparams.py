""" this script uses raw features instead of a pre-defined pipeline

features consist of:
- the probabilities from another media type classifier
    media_ocp/perceptron/cv2_lemma_1704443382.clf
- keyword list features
   datasets/ocp_entities_v0.csv

the featurizer does not get saved as part of the final .clf

in the case of OC keyword features this makes the model much smaller (by hundreds of MB)
"""
import json
import os
from os.path import dirname

from ovos_utils.log import LOG

from ovos_classifiers.skovos.features import KeywordFeaturesVectorizer
from ovos_classifiers.skovos.nn import PerceptronTrainer

csv_path = f"{dirname(dirname(__file__))}/datasets/ocp_sentences_v0.csv"
ents_csv_path = f"{dirname(dirname(__file__))}/datasets/ocp_entities_v0.csv"

clf_id = "binary_ocp_kw_tiny"


# the featurizer is also needed at inference
# it is excluded from pipeline to keep the final .clf model size small
# otherwise the ocp_kw pipeline adds hundreds of MB
class Featurizer:
    def __init__(self, ents):
        self.feats = KeywordFeaturesVectorizer()
        self.feats.load_entities(ents)

    def transform(self, X):
        return self.feats.transform(X)


def find_best_hyperparams_perceptron(n_searches=5):
    mkdown_table = """
| pipeline | language | accuracy | params | size (MB)|
|----------|----------|----------|--------|----------|"""

    os.makedirs(f"{dirname(__file__)}/classifiers/{clf_id}/perceptron", exist_ok=True)
    os.makedirs(f"{dirname(__file__)}/reports/{clf_id}/perceptron", exist_ok=True)

    featurizer = Featurizer(ents_csv_path)

    for i in range(n_searches):
        trainer = PerceptronTrainer("raw", featurizer)
        training_run = trainer.hyperparam_search(csv_path)
        n = f"{clf_id}_{str(training_run.end_ts).split('.')[0]}"
        model_path = f"{dirname(__file__)}/classifiers/{clf_id}/perceptron/{n}.clf"
        training_run.clf.save(model_path)
        report_path = f"{dirname(__file__)}/reports/{clf_id}/perceptron/{n}.txt"
        card_path = f"{dirname(__file__)}/reports/{clf_id}/perceptron/{n}.json"
        with open(card_path, "w") as f:
            json.dump({k: v for k, v in training_run.__dict__.items()
                       if isinstance(v, (str, int, float))}, f)
        with open(report_path, "w") as f:
            f.write(training_run.report)
        print(training_run.pipeline_id, training_run.hyperparams)
        print(training_run.report)

        size = round(os.path.getsize(model_path) / 1000000, 3)
        LOG.info(f"size {training_run.pipeline_id}.clf : {size} MB")

        lang = "all"
        mkdown_table += f"\n| {n} | {lang} | {training_run.accuracy}| {training_run.hyperparams} | {size} |"

        with open(f"{dirname(__file__)}/reports/{clf_id}/perceptron/bias_hyperparam_report.md", "w") as f:
            f.write(mkdown_table)

    print(mkdown_table)


find_best_hyperparams_perceptron()
