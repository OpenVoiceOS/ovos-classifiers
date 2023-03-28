import os

import joblib
from ovos_utils.xdg_utils import xdg_data_home
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

from ovos_classifiers.features.pipelines import get_features_pipeline


class OVOSClassifier:
    def __init__(self, pipeline_id, pipeline_clf):
        self.pipeline_id = pipeline_id.lower().split("-")[0]
        self._pipeline_clf = pipeline_clf
        self.clf = None

    def train(self, train_data, target_data):
        self.clf = Pipeline(self.pipeline)
        self.clf.fit(train_data, target_data)
        return self.clf

    @property
    def pipeline(self):
        return [
            ('features', get_features_pipeline(self.pipeline_id)),
            ('clf', self._pipeline_clf)
        ]

    def score(self, X, y):
        return self.clf.score(X, y)

    def predict(self, text):
        return self.clf.predict(text)

    def save(self, path):
        joblib.dump(self.clf, path)

    def load_from_file(self, path=None):
        if not path:
            os.makedirs(f"{xdg_data_home()}/OpenVoiceOS/classifiers", exist_ok=True)
            path = f"{xdg_data_home()}/OpenVoiceOS/classifiers/{self.pipeline_id}"
        self.clf = joblib.load(path)
        return self


class OVOSVotingClassifier(OVOSClassifier):
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
                ('features', get_features_pipeline(self.pipeline_id)),
                ('clf', clf)
            ])
        return pipes

    def train(self, train_data, target_data):
        for name, clf in self.voting_pipelines.items():
            print("training", name)
            clf.fit(train_data, target_data)
        print("training voting classifier")
        return super().train(train_data, target_data)
