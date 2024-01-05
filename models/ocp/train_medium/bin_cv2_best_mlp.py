import json
import os
from os.path import dirname

from ovos_utils.log import LOG

from ovos_classifiers.skovos.nn import MLPTrainer

csv_path = f"{dirname(dirname(__file__))}/datasets/ocp_sentences_v0.csv"
ents_csv_path = f"{dirname(dirname(__file__))}/datasets/ocp_entities_v0.csv"

clf_id = "binary_ocp_cv2_medium"
retrain = False
pipelines = [
    "cv2",
    "cv2_lemma",
    "skipgram2"
]


# select a pipeline, in our case we use the default OCP Keywords pipelines
# that uses the ocp_entities_v0.csv wordlist to extract features

def find_best_pipeline_mlp():
    global pipelines
    mkdown_table = """
| pipeline | language | accuracy | size (MB)|
|----------|----------|----------|----------|"""

    if not retrain:
        pipelines = [p for p in pipelines if not
        os.path.isfile(f"{dirname(__file__)}/reports/{clf_id}/mlp/{p}.json")]
    if not pipelines:
        return

    res = MLPTrainer.find_best_pipeline(pipelines,
                                        csv_path,
                                        search_hyperparams=False,
                                        threaded=False)

    os.makedirs(f"{dirname(__file__)}/classifiers/{clf_id}/mlp", exist_ok=True)
    os.makedirs(f"{dirname(__file__)}/reports/{clf_id}/mlp", exist_ok=True)

    for training_run in res:
        if training_run is None:
            continue
        LOG.info(training_run.pipeline_id)
        LOG.info(training_run.report)

        model_path = f"{dirname(__file__)}/classifiers/{clf_id}/mlp/{training_run.pipeline_id}.clf"
        # training_run.clf.save(model_path)

        report_path = f"{dirname(__file__)}/reports/{clf_id}/mlp/{training_run.pipeline_id}.txt"
        with open(report_path, "w") as f:
            f.write(training_run.report)
        card_path = f"{dirname(__file__)}/reports/{clf_id}/mlp/{training_run.pipeline_id}.json"
        with open(card_path, "w") as f:
            json.dump({k: v for k, v in training_run.__dict__.items()
                       if isinstance(v, (str, int, float))}, f)

        size = 0
        # size = round(os.path.getsize(model_path) / 1000000, 3)
        LOG.info(f"size {training_run.pipeline_id}.clf : {size} MB")
        lang = "en"
        mkdown_table += f"\n| {training_run.pipeline_id} | {lang} | {training_run.accuracy}| {size} |"

        with open(f"{dirname(__file__)}/reports/{clf_id}/mlp/report.md", "w") as f:
            f.write(mkdown_table)

        print(mkdown_table)


def find_best_hyperparams_mlp(pipeline_id, n_searches=3):
    mkdown_table = """
| pipeline | language | accuracy | params | size (MB)|
|----------|----------|----------|--------|----------|"""
    os.makedirs(f"{dirname(__file__)}/classifiers/{clf_id}/mlp", exist_ok=True)
    os.makedirs(f"{dirname(__file__)}/reports/{clf_id}/mlp", exist_ok=True)

    for i in range(n_searches):
        trainer = MLPTrainer(pipeline_id)
        training_run = trainer.hyperparam_search(csv_path)
        n = f"{training_run.pipeline_id}_{str(training_run.end_ts).split('.')[0]}"
        model_path = f"{dirname(__file__)}/classifiers/{clf_id}/mlp/{n}.clf"
        training_run.clf.save(model_path)
        report_path = f"{dirname(__file__)}/reports/{clf_id}/mlp/{n}.txt"
        card_path = f"{dirname(__file__)}/reports/{clf_id}/mlp/{n}.json"
        with open(card_path, "w") as f:
            json.dump({k: v for k, v in training_run.__dict__.items()
                       if isinstance(v, (str, int, float))}, f)
        with open(report_path, "w") as f:
            f.write(training_run.report)
        print(training_run.pipeline_id, training_run.hyperparams)
        print(training_run.report)

        size = round(os.path.getsize(model_path) / 1000000, 3)
        LOG.info(f"size {training_run.pipeline_id}.clf : {size} MB")

        lang = "en"
        mkdown_table += f"\n| {n} | {lang} | {training_run.accuracy}| {training_run.hyperparams} | {size} |"

        with open(f"{dirname(__file__)}/reports/{clf_id}/mlp/{pipeline_id}_hyperparam_report.md", "w") as f:
            f.write(mkdown_table)

    print(mkdown_table)


#find_best_pipeline_mlp()

find_best_hyperparams_mlp("cv2")  # tiny
find_best_hyperparams_mlp("cv2_lemma")  # tiny
find_best_hyperparams_mlp("skipgram2")  # tiny
