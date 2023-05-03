import json
from os import makedirs
from os.path import isfile
import nltk
import requests
from ovos_config import Configuration
from ovos_utils.xdg_utils import xdg_data_home

from ovos_classifiers.heuristics.postag import NltkPostag, RegexPostag
from ovos_classifiers.utils import load_tagger

# TODO - benchmark and choose based on performance/model size
# TODO - ensure all langs have 1 model
_LANGDEFAULTS = {
    "pt": "nltk-floresta+mcmorpho-udep-brill-postag",
    "es": "nltk-cess_esp-udep-brill-postag",
    "ca": "nltk-cess_cat-udep-brill-postag",
    "en": "nltk"
}


class OVOSPostag:
    _XDG_PATH = f"{xdg_data_home()}/OpenVoiceOS/classifiers"
    _BASE_METADATA_URL = "https://github.com/OpenVoiceOS/ovos-classifiers/raw/dev/models/metadata"
    _BASE_MODEL_URL = "https://github.com/OpenVoiceOS/ovos-classifiers/raw/dev/models/postag"
    makedirs(_XDG_PATH, exist_ok=True)

    def __init__(self, model_id=None, lang=None):
        config_core = Configuration()
        self.lang = lang or config_core.get("lang", "en-us")
        self.config = config_core.get("classifiers", {}).get("postag", {})
        model_id = model_id or self.config.get("model_id") or self.lang.split("-")[0]
        if model_id in _LANGDEFAULTS:
            model_id = _LANGDEFAULTS.get(model_id)
        self.model_id = model_id
        self.meta, self.clf = self.load_model(self.model_id)

    @property
    def tagset(self):
        return self.meta.get("tagset")

    def get_model(self, model_id):
        if model_id in _LANGDEFAULTS:
            model_id = _LANGDEFAULTS.get(model_id)

        if model_id == "nltk":
            return {"model_id": "nltk",
                    "tagset": "Universal Dependencies",
                    "lang": self.lang,
                    "algo": "heuristic"}, NltkPostag
        elif model_id == "regex":
            return {"model_id": "regex",
                    "tagset": "Universal Dependencies",
                    "lang": self.lang,
                    "algo": "heuristic"}, RegexPostag

        meta_path = f"{self._XDG_PATH}/{model_id}.json"
        if isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            url = f"{self._BASE_METADATA_URL}/{model_id}.json"
            meta = requests.get(url).json()
            with open(meta_path, "wb") as f:
                f.write(meta)

        model_path = f"{self._XDG_PATH}/{model_id}.pkl"
        if not isfile(model_path):
            url = f"{self._BASE_MODEL_URL}/{model_id}.pkl"
            model = requests.get(url).content
            with open(model_path, "wb") as f:
                f.write(model)

        return meta, model_path

    def load_model(self, model_id):
        if model_id == "nltk":
            nltk.download("punkt")
            nltk.download('averaged_perceptron_tagger')
            nltk.download('universal_tagset')
        data, model_path = self.get_model(model_id)
        return load_tagger(data, model_path)

    def postag(self, sentence):
        return self.clf.tag(sentence)


if __name__ == "__main__":
    p = OVOSPostag("regex")
    print(p.model_id)
    print(p.tagset)
    print(p.postag("The brown fox jumped over the lazy dog"))
    # [('The', 'DET'), ('brown', 'ADJ'), ('fox', 'NOUN'), ('jumped', 'VERB'), ('over', 'ADP'), ('the', 'DET'), ('lazy', 'ADJ'), ('dog', 'NOUN')]


    p = OVOSPostag("nltk")
    print(p.model_id)
    print(p.tagset)
    print(p.postag("The brown fox jumped over the lazy dog"))
    # [('The', 'DET'), ('brown', 'ADJ'), ('fox', 'NOUN'), ('jumped', 'VERB'), ('over', 'ADP'), ('the', 'DET'), ('lazy', 'ADJ'), ('dog', 'NOUN')]

    p = OVOSPostag("regex", "pt")
    print(p.model_id)
    print(p.tagset)
    print(p.postag("Ontem eu fui passear com o meu cão"))
    # [('Ontem', ('Ontem', 'ADV')), ('fui', ('fui', 'VERB')), ('passear', ('passear', 'VERB')), ('com', ('com', 'ADP')), ('o', ('o', 'DET')), ('meu', ('meu', 'PRON')), ('cão', ('cão', 'NOUN'))]

    p = OVOSPostag("pt")
    print(p.model_id)
    print(p.tagset)
    print(p.postag("Ontem fui passear com o meu cão"))
    # [('Ontem', ('Ontem', 'ADV')), ('fui', ('fui', 'VERB')), ('passear', ('passear', 'VERB')), ('com', ('com', 'ADP')), ('o', ('o', 'DET')), ('meu', ('meu', 'PRON')), ('cão', ('cão', 'NOUN'))]

    p = OVOSPostag("nltk-brown-brown-ngram-postag")
    print(p.model_id)
    print(p.tagset)
    print(p.postag("The brown fox jumped over the lazy dog"))
    # [('The', ('The', 'AT')), ('brown', ('brown', 'JJ')), ('fox', ('fox', 'NN')), ('jumped', ('jumped', 'VBD')), ('over', ('over', 'IN')), ('the', ('the', 'AT')), ('lazy', ('lazy', 'JJ')), ('dog', ('dog', 'NN'))]

    p = OVOSPostag("nltk-floresta-visl-brill-postag")
    print(p.model_id)
    print(p.tagset)
    print(p.postag("Ontem fui passear com o meu cão"))
    # [('Ontem', ('Ontem', 'adv')), ('fui', ('fui', 'v-fin')), ('passear', ('passear', 'v-inf')), ('com', ('com', 'prp')), ('o', ('o', 'art')), ('meu', ('meu', 'pron-det')), ('cão', ('cão', 'n'))]
