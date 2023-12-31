import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from ovos_classifiers.skovos.pipelines import get_features_pipeline
from ovos_classifiers.tasks.classifier import OVOSAbstractClassifier


def iter_clfs(calibrate=True, feature_select=False, voting=False):
    if voting:
        clfs = {

            "v_clsvccpcmlp": VotingClassifier(
                estimators=[("CalibratedLSVC", CalibratedClassifierCV(LinearSVC(dual="auto", penalty="l1"))),
                            ("CalibratedPerceptron", CalibratedClassifierCV(Perceptron())),
                            ("CalibratedMLP", MLPClassifier(max_iter=1000)),
                            ], voting="soft"),
            "v_cpcmlp": VotingClassifier(
                estimators=[("CalibratedPerceptron", CalibratedClassifierCV(Perceptron())),
                            ("CalibratedMLP", MLPClassifier(max_iter=1000)),
                            ], voting="soft")
        }
    elif feature_select:
        clfs = {
            "fs_lsvc_p": Pipeline([
                ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1"))),
                ('classification', Perceptron())
            ]),
            "fs_lsvc_lr": Pipeline([
                ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1"))),
                ('classification', LogisticRegression())
            ]),
            "fs_lsvc_svc": Pipeline([
                ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1"))),
                ('classification', SVC(kernel='linear', probability=True))
            ]),
            "fs_lsvc_mlp": Pipeline([
                ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1"))),
                ('classification', MLPClassifier(max_iter=300))
            ])
        }
    else:
        clfs = {
            "lsvc": LinearSVC(),
            "percep": Perceptron(),
            "pa": PassiveAggressiveClassifier(),
            "et": ExtraTreesClassifier(n_estimators=50),
            "mlp": MLPClassifier(max_iter=1000),
            "svc": SVC(kernel='linear', probability=True),
            "lr": LogisticRegression(),
            "mnb": MultinomialNB(),
            "gnb": GaussianNB(),
            "bnb": BernoulliNB(),
            "rf": RandomForestClassifier()
        }

    if calibrate:
        clfs = {f"c_{k}": CalibratedClassifierCV(clf)
                for k, clf in clfs.items()}

    yield from clfs.items()


class SklearnOVOSClassifier(OVOSAbstractClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pipeline_id == "raw":
            self.clf = self._pipeline_clf

    def train(self, train_data, target_data):
        if self.pipeline_id != "raw":
            self.clf = Pipeline(self.pipeline)
        else:
            self.clf = self._pipeline_clf
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('utils', get_features_pipeline(self.pipeline_id)),
            ('clf', self._pipeline_clf)
        ]

    def score(self, X, y):
        return self.clf.score(X, y)

    def predict(self, text):
        return self.clf.predict(text)

    def predict_proba(self, text):
        return np.max(self.clf.predict_proba(text), axis=1)

    def predict_labels(self, utterances):
        probs = []
        for pred in self.clf.predict_proba(utterances):
            classes = self.clf.classes_
            labels = {classes[idx]: p for idx, p in enumerate(pred)}
            probs.append(labels)
        return probs


class SklearnOVOSVotingClassifier(SklearnOVOSClassifier):
    def __init__(self, voter_clfs, pipeline_id, voting='soft', weights=None):

        # sklearn style
        # voter_clfs = [('dt', clf1), ('knn', clf2), ('svc', clf3)]
        # ovos style
        # voter_clfs = [¢lf1, clf2, clf3]
        if not isinstance(voter_clfs[0], tuple):
            voter_clfs = [(c.__class__.__name__, c) for c in voter_clfs]
        self.voter_clfs = voter_clfs

        pipeline_clf = VotingClassifier(estimators=self.voter_clfs, voting=voting, weights=weights)
        super().__init__(pipeline_id=pipeline_id, pipeline_clf=pipeline_clf)

    @property
    def voting_pipelines(self):
        pipes = {}
        for name, clf in self.voter_clfs:
            pipes[name] = Pipeline([
                ('utils', get_features_pipeline(self.pipeline_id)),
                ('clf', clf)
            ])
        return pipes

    def train(self, train_data, target_data):
        for name, clf in self.voting_pipelines.items():
            print("training", name)
            clf.fit(train_data, target_data)
        print("training voting classifier")
        return super().train(train_data, target_data)

    def predict_proba(self, text):
        return np.max(self.clf.predict_proba(text), axis=1)
