from ovos_classifiers.skovos.classifier import SklearnOVOSVotingClassifier, SklearnOVOSClassifier
from ovos_classifiers.tasks.tagger import OVOSAbstractClassifierTagger


class SklearnOVOSClassifierTagger(SklearnOVOSClassifier, OVOSAbstractClassifierTagger):
    def __init__(self, pipeline_clf=None, pipeline_id="naive"):
        super().__init__(pipeline_clf=pipeline_clf, pipeline_id=pipeline_id)

    def score(self, X_test, y_test):
        return self.clf.score(X_test, y_test)


class SklearnOVOSVotingClassifierTagger(SklearnOVOSVotingClassifier, OVOSAbstractClassifierTagger):
    def __init__(self, voter_clfs, pipeline_id="naive", voting='hard', weights=None):
        super().__init__(voter_clfs, pipeline_id, voting, weights)

    def score(self, X_test, y_test):
        return self.clf.score(X_test, y_test)
