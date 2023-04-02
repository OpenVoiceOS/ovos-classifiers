import random
from os import makedirs
from os.path import isfile

import nltk
import requests
from nltk.corpus import treebank
from ovos_utils.xdg_utils import xdg_data_home


def _tagged_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(tagged[index][0])
            y.append(tagged[index][1])

    return X, y


# corefiob_v0.1
def get_coref_iob_tagged_sents():
    base_path = f"{xdg_data_home()}/OpenVoiceOS/datasets"
    makedirs(base_path, exist_ok=True)
    path = f"{base_path}/corefiob_v0.1.txt"
    url = "https://github.com/OpenVoiceOS/ovos-datasets/raw/master/text/corefiob_v0.1.txt"
    if not isfile(path):
        data = requests.get(url).text
        with open(path, "w") as f:
            f.write(data)

    corpus = []
    with open(path) as f:
        for l in f.read().split("\n\n"):
            l = l.replace(" ", "\t")
            toks = [(t.split("\t")[0], t.split("\t")[2])
                    for t in l.split("\n") if t.strip()]
            corpus.append(toks)

    return corpus


def get_coref_iob_dataset():
    tagged = get_coref_iob_tagged_sents()
    return _tagged_to_dataset(tagged)


def get_coref_iob_trainset():
    utts, tags = get_coref_iob_dataset()
    t = int(len(utts) * 0.8)
    X = utts[:t]
    y = tags[:t]
    X_test = utts[t:]
    y_test = tags[t:]
    return (X, y), (X_test, y_test)


# world_names_v0.2.csv
def get_world_names_tagged_words():
    base_path = f"{xdg_data_home()}/OpenVoiceOS/datasets"
    makedirs(base_path, exist_ok=True)
    path = f"{base_path}/world_names_v0.2.csv"
    url = "https://github.com/OpenVoiceOS/ovos-datasets/raw/master/text/world_names_v0.2.csv"
    if not isfile(path):
        data = requests.get(url).text
        with open(path, "w") as f:
            f.write(data)

    corpus = []
    with open(path) as f:
        for l in f.read().split("\n")[1:]:
            tag, name, lang = l.split(",", 2)
            corpus.append((name, tag))

    return corpus


def get_world_names_dataset():
    tagged = get_world_names_tagged_words()
    return _tagged_to_dataset([tagged])


def get_world_names_trainset():
    utts, tags = get_world_names_dataset()
    t = int(len(utts) * 0.8)
    X = utts[:t]
    y = tags[:t]
    X_test = utts[t:]
    y_test = tags[t:]
    return (X, y), (X_test, y_test)



# utterance_tags_v0.1
def get_utterance_tags_tagged_sents():
    base_path = f"{xdg_data_home()}/OpenVoiceOS/datasets"
    makedirs(base_path, exist_ok=True)
    path = f"{base_path}/utterance_tags_v0.1.csv"
    url = "https://github.com/OpenVoiceOS/ovos-datasets/raw/master/text/utterance_tags_v0.1.csv"
    if not isfile(path):
        data = requests.get(url).text
        with open(path, "w") as f:
            f.write(data)

    corpus = []
    with open(path) as f:
        for l in f.read().split("\n")[1:]:
            utt, tag = l.split(",", 1)[::-1]
            corpus.append((utt, tag))

    return corpus


def get_utterance_tags_dataset():
    tagged = get_utterance_tags_tagged_sents()
    return _tagged_to_dataset(tagged)


def get_utterance_tags_trainset():
    utts, tags = get_utterance_tags_dataset()
    t = int(len(utts) * 0.8)
    X = utts[:t]
    y = tags[:t]
    X_test = utts[t:]
    y_test = tags[t:]
    return (X, y), (X_test, y_test)


# Treebank
def get_treebank_tagged_sents(udep=False):
    nltk.download('treebank')
    if udep:
        corpus = list(treebank.tagged_sents(tagset="universal"))
    else:
        corpus = list(treebank.tagged_sents())
    return corpus


def get_treebank_dataset(udep=False):
    corpus = get_treebank_tagged_sents(udep)
    return _tagged_to_dataset(corpus)


def get_treebank_trainset(udep=False):
    corpus = get_treebank_tagged_sents(udep)
    random.shuffle(corpus)
    train_data = corpus[:3000]
    test_data = corpus[3000:]
    X, y = _tagged_to_dataset(train_data)
    X_test, y_test = _tagged_to_dataset(test_data)
    return (X, y), (X_test, y_test)


# Brown
def get_brown_tagged_sents(udep=False):
    nltk.download('treebank')
    if udep:
        corpus = list(treebank.tagged_sents(tagset="universal"))
    else:
        corpus = list(treebank.tagged_sents())
    return corpus


def get_brown_dataset(udep=False):
    corpus = get_treebank_tagged_sents(udep)
    return _tagged_to_dataset(corpus)


def get_brown_trainset(udep=False):
    corpus = get_treebank_tagged_sents(udep)
    random.shuffle(corpus)
    train_data = corpus[:3000]
    test_data = corpus[3000:]
    X, y = _tagged_to_dataset(train_data)
    X_test, y_test = _tagged_to_dataset(test_data)
    return (X, y), (X_test, y_test)
