import abc
import random
import time
from dataclasses import dataclass

from ovos_utils.log import LOG
from ovos_utils import create_daemon
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier
from ovos_classifiers.skovos.pipelines import get_features_pipeline


@dataclass
class TrainingRun:
    pipeline_id: str
    hyperparams: dict
    clf: SklearnOVOSClassifier
    accuracy: float = 0
    report: str = ""
    start_ts: float = 0
    end_ts: float = 0


class BaseTrainer:

    def __init__(self, pipeline_id="raw", featurizer=None):
        self.pipeline_id = pipeline_id
        self.featurizer = featurizer

    def split_train_test(self, csv_path, test_size=0.6):
        X, y = self.read_csv(csv_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            stratify=y)
        return X_train, X_test, y_train, y_test

    def read_csv(self, csv_paths):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        data = []
        for csv_path in csv_paths:
            with open(csv_path) as f:
                lines = f.readlines()[1:]
                data += [line.strip().split(",") for line in lines if "," in line]
        X = [line[1] for line in data]
        y = [line[0] for line in data]
        return X, y

    @abc.abstractmethod
    def hyperparam_search(self, csv_path, test_csv_path=None, parameter_space=None) -> TrainingRun:
        raise NotImplemented

    @abc.abstractmethod
    def train(self, csv_path, test_csv_path=None) -> TrainingRun:
        raise NotImplemented

    @classmethod
    def find_best_pipeline(cls, pipelines, csv_path, test_csv_path=None,
                           search_hyperparams=False, threaded=False, n_threads=4):
        clfs = {p: None for p in pipelines}
        trainers = []

        for pipeline_id in pipelines:

            # find the best hyperparams for a classifier with this pipeline
            def train(pid=pipeline_id):
                trainer = cls(pid)
                if search_hyperparams:
                    LOG.info(f"finding best hyperparams for pipeline: {pid}")
                    clfs[pid] = trainer.hyperparam_search(csv_path, test_csv_path)
                else:
                    LOG.info(f"training pipeline: {pid}")
                    clfs[pid] = trainer.train(csv_path, test_csv_path)

            train.pid = pipeline_id

            trainers.append(train)

        # start training
        threads = []
        for t in trainers:
            if threaded:
                # 1 thread for each pipeline id
                if len(threads) < n_threads:
                    tr = create_daemon(t)
                    tr.pipeline_id = t.pid
                    threads.append(tr)
                else:
                    threads[0].join()  # wait
                    yield clfs[threads[0].pipeline_id]
                    threads = threads[1:]
            else:
                t()
                yield clfs[t.pid]

        # wait for all pipelines to train
        for t in threads:
            t.join()
            yield clfs[t.pipeline_id]


class MLPTrainer(BaseTrainer):

    def train(self, csv_path, test_csv_path=None, max_iter=100, calibrate=True) -> TrainingRun:
        start_ts = time.time()
        if test_csv_path:
            X_train, y_train = self.read_csv(csv_path)
            X_test, y_test = self.read_csv(test_csv_path)
        else:
            X_train, X_test, y_train, y_test = self.split_train_test(csv_path, test_size=0.6)

        c = MLPClassifier(max_iter=max_iter,
                          verbose=True)
        if calibrate:
            c = CalibratedClassifierCV(c)

        clf = SklearnOVOSClassifier(self.pipeline_id, c)
        clf.train(X_train, y_train)

        # test the classifier
        # Note that we dont feed features here,
        # the calibrated classifier uses the pipeline_id internally
        y_pred = clf.predict(X_test)
        acc = balanced_accuracy_score(y_test, y_pred)
        report = f"Balanced Accuracy: {acc}\n" + \
                 classification_report(y_test, y_pred, target_names=c.classes_)
        LOG.info(f"{self.pipeline_id} Accuracy: {acc}")

        run = TrainingRun(
            pipeline_id=self.pipeline_id,
            hyperparams={},
            clf=clf,
            accuracy=acc,
            report=report,
            start_ts=start_ts,
            end_ts=time.time()
        )

        return run

    def hyperparam_search(self, csv_path, test_csv_path=None, max_iter=100,
                          parameter_space=None, n_jobs=3, test_size=0.6) -> TrainingRun:

        start_ts = time.time()
        if test_csv_path:
            X_train, y_train = self.read_csv(csv_path)
            X_test, y_test = self.read_csv(test_csv_path)
        else:
            X_train, X_test, y_train, y_test = self.split_train_test(csv_path, test_size=test_size)

        if self.pipeline_id == "raw":
            pipeline = self.featurizer
        else:
            pipeline = get_features_pipeline(self.pipeline_id)
            pipeline.fit(X_train, y_train)  # train/prepare feature extractors

        # define random search params
        parameter_space = parameter_space or {
            'hidden_layer_sizes': [(random.randint(10, 80), random.randint(80, 150)),
                                   (random.randint(50, 150), random.randint(20, 50)),
                                   (random.randint(20, 150), random.randint(20, 150)),
                                   (random.randint(100, 250),),
                                   (120, 20, 80),
                                   (random.randint(20, 150), random.randint(20, 150), random.randint(20, 150)),
                                   (random.randint(100, 150), random.randint(20, 150), random.randint(20, 50)),
                                   (random.randint(20, 50), random.randint(50, 150), random.randint(20, 150))],
            'activation': ["identity", "logistic", "tanh", "relu"],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'early_stopping': [True, False],
            'alpha': [
                0.001 * random.randint(1, 10),
                0.0005 * random.randint(1, 100),
                0.01 * random.randint(1, 10),
                0.05],
            'learning_rate': ['constant', 'adaptive', 'invscaling'],
        }
        mlp_gs = MLPClassifier(max_iter=max_iter,
                               verbose=True)

        # do a random search
        c = RandomizedSearchCV(mlp_gs, parameter_space, n_jobs=n_jobs, cv=5)
        feats = pipeline.transform(X_train)  # run trough pipeline feature extractor

        c.fit(feats, y_train)

        LOG.info(f'Best parameters found:\n {c.best_params_}')

        # calibrate the classifier
        # we want the output to be directly interpretable as a probability
        LOG.info("Calibrating classifier")
        calibrated = CalibratedClassifierCV(c.best_estimator_)
        clf = SklearnOVOSClassifier(self.pipeline_id, calibrated)
        if self.pipeline_id == "raw":
            clf.train(feats, y_train)
            feats = pipeline.transform(X_test)
            y_pred = clf.predict(feats)
        else:
            clf.train(X_train, y_train)
            y_pred = clf.predict(X_test)

        # test the classifier
        # Note that we dont feed features here,
        # the calibrated classifier uses the pipeline_id internally
        acc = balanced_accuracy_score(y_test, y_pred)

        report = f"Balanced Accuracy: {acc}\n" + \
                 classification_report(y_test, y_pred, target_names=c.classes_)
        LOG.info(f"{self.pipeline_id} Accuracy: {acc}")

        run = TrainingRun(
            pipeline_id=self.pipeline_id,
            hyperparams=c.best_params_,
            clf=clf,
            accuracy=acc,
            report=report,
            start_ts=start_ts,
            end_ts=time.time()
        )

        return run


class PerceptronTrainer(BaseTrainer):

    def train(self, csv_path, test_csv_path=None,  calibrate=True) -> TrainingRun:
        start_ts = time.time()
        if test_csv_path:
            X_train, y_train = self.read_csv(csv_path)
            X_test, y_test = self.read_csv(test_csv_path)
        else:
            X_train, X_test, y_train, y_test = self.split_train_test(csv_path, test_size=0.6)

        if self.pipeline_id == "raw":
            pipeline = self.featurizer
        else:
            pipeline = get_features_pipeline(self.pipeline_id)
            pipeline.fit(X_train, y_train)  # train/prepare feature extractors

        c = Perceptron( verbose=True)
        if calibrate:
            c = CalibratedClassifierCV(c)

        clf = SklearnOVOSClassifier(self.pipeline_id, c)

        if self.pipeline_id == "raw":
            feats = pipeline.transform(X_train)  # run trough pipeline feature extractor
            clf.train(feats, y_train)
            feats = pipeline.transform(X_test)
            y_pred = clf.predict(feats)
        else:
            clf.train(X_train, y_train)
            y_pred = clf.predict(X_test)

        # test the classifier
        # Note that we dont feed features here,
        # the calibrated classifier uses the pipeline_id internally
        acc = balanced_accuracy_score(y_test, y_pred)
        report = f"Balanced Accuracy: {acc}\n" + \
                 classification_report(y_test, y_pred, target_names=c.classes_)
        LOG.info(f"{self.pipeline_id} Accuracy: {acc}")

        run = TrainingRun(
            pipeline_id=self.pipeline_id,
            hyperparams={},
            clf=clf,
            accuracy=acc,
            report=report,
            start_ts=start_ts,
            end_ts=time.time()
        )

        return run

    def hyperparam_search(self, csv_path, test_csv_path=None,
                          parameter_space=None, n_jobs=3,
                          test_size=0.6) -> TrainingRun:

        start_ts = time.time()
        if test_csv_path:
            X_train, y_train = self.read_csv(csv_path)
            X_test, y_test = self.read_csv(test_csv_path)
        else:
            X_train, X_test, y_train, y_test = self.split_train_test(csv_path, test_size=test_size)

        if self.pipeline_id == "raw":
            pipeline = self.featurizer
        else:
            pipeline = get_features_pipeline(self.pipeline_id)
            pipeline.fit(X_train, y_train)  # train/prepare feature extractors

        # define random search params
        parameter_space = parameter_space or {
            'penalty': ["l2", "l1", "elasticnet", None],
            'alpha': [0.0001, 0.002, 0.005, 0.01, 0.02, 0.07, 0.1, 0.05],
            'l1_ratio': [0.15, 0.3, 0.5, 0.7, 0.9],
            'early_stopping': [True, False]
        }
        mlp_gs = Perceptron(verbose=True)

        # do a random search
        c = RandomizedSearchCV(mlp_gs, parameter_space, n_jobs=n_jobs, cv=5)
        feats = pipeline.transform(X_train)  # run trough pipeline feature extractor

        c.fit(feats, y_train)

        LOG.info(f'Best parameters found:\n {c.best_params_}')

        # calibrate the classifier
        # we want the output to be directly interpretable as a probability
        LOG.info("Calibrating classifier")
        calibrated = CalibratedClassifierCV(c.best_estimator_)
        clf = SklearnOVOSClassifier(self.pipeline_id, calibrated)
        if self.pipeline_id == "raw":
            clf.train(feats, y_train)
            feats = pipeline.transform(X_test)
            y_pred = clf.predict(feats)
        else:
            clf.train(X_train, y_train)
            y_pred = clf.predict(X_test)

        # test the classifier
        # Note that we dont feed features here,
        # the calibrated classifier uses the pipeline_id internally
        acc = balanced_accuracy_score(y_test, y_pred)

        report = f"Balanced Accuracy: {acc}\n" + \
                 classification_report(y_test, y_pred, target_names=c.classes_)
        LOG.info(f"{self.pipeline_id} Accuracy: {acc}")

        run = TrainingRun(
            pipeline_id=self.pipeline_id,
            hyperparams=c.best_params_,
            clf=clf,
            accuracy=acc,
            report=report,
            start_ts=start_ts,
            end_ts=time.time()
        )

        return run
