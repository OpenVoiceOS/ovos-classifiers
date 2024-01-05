import json
import os
from os.path import dirname

from ovos_utils.log import LOG

from ovos_classifiers.skovos.nn import MLPTrainer, PerceptronTrainer

csv_small_path = f"{dirname(__file__)}/datasets/ocp_media_types_balanced_small_v0.csv"
csv_path = f"{dirname(__file__)}/datasets/ocp_media_types_v0.csv"
csv_big_path = f"{dirname(__file__)}/datasets/ocp_media_types_balanced_big_v0.csv"

clf_id = "media_ocp"
retrain = False


# select a pipeline, in our case we use the default OCP Keywords pipelines
# that uses the ocp_entities_v0.csv wordlist to extract features

def find_best_pipeline_mlp(pipelines=None):
    mkdown_table = """
| pipeline | language | accuracy | size (MB)|
|----------|----------|----------|----------|"""

    pipelines = pipelines or [
        #"ocp_kw",  # 0.8790252196742979  <- lang agnostic, but needs keyword list

        #"ocp_kw_cv2",  # 0.9837252065904116 <- english only, needs keyword list
        #"ocp_kw_tfidf",  # 0.926142905877921 <- english only, needs keyword list
        #"ocp_kw_cv2_lemma",  # 0.9812331690578716 <- english only, needs keyword list
        #"ocp_kw_tfidf_lemma",  # 0.9213230895646087  <- english only, needs keyword list

        "cv2",  # 0.9855151519448357 <- english only
        "cv2_lemma",  # 0.9874539948380118 <- english only **BEST
        "skipgram2",  # 0.9666248225791833 <- english only
        "tfidf",  # 0.838108187960996 <- english only
        "tfidf_lemma",  # 0.7857186695170012 <- english only

    ]
    if not retrain:
        pipelines = [p for p in pipelines if not
                     os.path.isfile(f"{dirname(__file__)}/reports/{clf_id}/mlp/{p}.json")]
    if not pipelines:
        return

    res = MLPTrainer.find_best_pipeline(pipelines,
                                               csv_path,
                                               test_csv_path=csv_big_path,
                                               search_hyperparams=False,
                                               threaded=True)

    os.makedirs(f"{dirname(__file__)}/classifiers/{clf_id}/mlp", exist_ok=True)
    os.makedirs(f"{dirname(__file__)}/reports/{clf_id}/mlp", exist_ok=True)


    for training_run in res:
        if training_run is None:
            continue
        LOG.info(training_run.pipeline_id)
        LOG.info(training_run.report)

        model_path = f"{dirname(__file__)}/classifiers/{clf_id}/mlp/{training_run.pipeline_id}.clf"
        #training_run.clf.save(model_path)

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
        lang = "en" if training_run.pipeline_id != "ocp_kw" else "all"
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
        training_run = trainer.hyperparam_search(csv_path, test_csv_path=csv_big_path)
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

        lang = "en" if training_run.pipeline_id != "ocp_kw" else "all"
        mkdown_table += f"\n| {n} | {lang} | {training_run.accuracy}| {training_run.hyperparams} | {size} |"

        with open(f"{dirname(__file__)}/reports/{clf_id}/mlp/{pipeline_id}_hyperparam_report.md", "w") as f:
            f.write(mkdown_table)

    print(mkdown_table)


find_best_pipeline_mlp()


find_best_hyperparams_mlp("cv2_lemma")  # small
