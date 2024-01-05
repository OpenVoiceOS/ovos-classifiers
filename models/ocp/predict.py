from os.path import dirname

from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier

# tiny
path = f"{dirname(__file__)}/classifiers/binary_ocp/perceptron/tfidf_lemma_1704442719.clf"
# small
path = f"{dirname(__file__)}/classifiers/binary_ocp/perceptron/cv2_lemma_1704442563.clf"


utterances = ["play a song", "play my morning jams",
              "i want to watch the matrix",
              "tell me a joke", "who are you", "you suck"]

clf = SklearnOVOSClassifier.from_file(path)

preds = clf.predict(utterances)
print(preds)
# ['OCP' 'OCP' 'OCP' 'other' 'other' 'other']

for preds in clf.predict_labels(utterances):
    print(preds)
    # {'OCP': 0.9963047671984526, 'other': 0.0036952328015473573}
    # {'OCP': 0.784891933615923, 'other': 0.21510806638407692}
    # {'OCP': 0.8484310964958078, 'other': 0.1515689035041922}
    # {'OCP': 0.14793627487168542, 'other': 0.8520637251283144}
    # {'OCP': 0.00012829987903095752, 'other': 0.9998717001209692}
    # {'OCP': 0.036421046297418026, 'other': 0.9635789537025821}


path = f"{dirname(__file__)}/classifiers/media_ocp/perceptron/cv2_lemma_1704443382.clf"

utterances = [
    "play metallica", "play internet radio", "watch kill bill", "turn on the radio"
]

clf = SklearnOVOSClassifier.from_file(path)
preds = clf.predict(utterances)
print(preds)
# ['music' 'radio' 'movie' 'radio']


path = f"{dirname(__file__)}/classifiers/playback_ocp/perceptron/cv2_lemma_1704442616.clf"
clf = SklearnOVOSClassifier.from_file(path)
for preds in clf.predict_labels(utterances):
    print(preds)
    # {'audio': 0.5306805263015526, 'external': 0.07380390321730261, 'video': 0.3955155704811447}
    # {'audio': 0.9367481352739745, 'external': 0.055747939198030706, 'video': 0.007503925527994946}
    # {'audio': 0.023420751398999927, 'external': 0.05313549542410466, 'video': 0.9234437531768954}
    # {'audio': 0.9197664004106791, 'external': 0.05695515594079632, 'video': 0.023278443648524626}

from ovos_classifiers.skovos.features import ClassifierProbaVectorizer

v = ClassifierProbaVectorizer(clf)
print(v.transform(utterances))
# [[0.53068053 0.0738039  0.39551557]
#  [0.93674814 0.05574794 0.00750393]
#  [0.02342075 0.0531355  0.92344375]
#  [0.9197664  0.05695516 0.02327844]]
