import os
import abc
import joblib
from ovos_utils.xdg_utils import xdg_data_home


class OVOSAbstractClassifier:
    def __init__(self, pipeline_id, pipeline_clf):
        self.pipeline_id = pipeline_id.lower().split("-")[0]
        self._pipeline_clf = pipeline_clf
        self.clf = None

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        clf = cls(*args, **kwargs)
        clf.load_from_file(path)
        # TODO - self._pipeline_clf is None
        # TODO - self._pipeline_id is "naive"
        clf._retrainable = False
        return clf

    @abc.abstractmethod
    def train(self, train_data, target_data):
        raise NotImplemented

    @property
    def pipeline(self):
        return []

    @abc.abstractmethod
    def score(self, X, y):
        raise NotImplemented

    @abc.abstractmethod
    def predict(self, text):
        raise NotImplemented

    def save(self, path):
        joblib.dump(self.clf, path)

    def load_from_file(self, path=None):
        if not path:
            os.makedirs(f"{xdg_data_home()}/OpenVoiceOS/classifiers", exist_ok=True)
            path = f"{xdg_data_home()}/OpenVoiceOS/classifiers/{self.pipeline_id}"
        self.clf = joblib.load(path)
        return self

