import json
from os import makedirs
from os.path import isfile

import requests
from ovos_config import Configuration
from ovos_utils.xdg_utils import xdg_data_home

# TODO - benchmark and choose based on performance/model size
# TODO - ensure all langs have 1 model
_LANGDEFAULTS = {
    "en": "sklearn-utterance_tags_v0.1-utttags-lsvc-clf"
}


class OVOSUtteranceTagger:
    _XDG_PATH = f"{xdg_data_home()}/OpenVoiceOS/classifiers"
    _BASE_METADATA_URL = "https://github.com/OpenVoiceOS/ovos-classifiers/raw/dev/models/metadata"
    _BASE_MODEL_URL = "https://github.com/OpenVoiceOS/ovos-classifiers/raw/dev/models/utttags"
    makedirs(_XDG_PATH, exist_ok=True)

    def __init__(self, model_id=None):
        config_core = Configuration()
        lang = config_core.get("lang", "en-us")
        self.config = config_core.get("classifiers", {}).get("postag", {})
        model_id = model_id or self.config.get("model_id") or lang.split("-")[0]
        if model_id in _LANGDEFAULTS:
            model_id = _LANGDEFAULTS.get(model_id)
        self.model_id = model_id
        self.meta, self.clf = self.load_model(self.model_id)

    @property
    def tagset(self):
        return self.meta.get("tagset")

    @classmethod
    def get_model(cls, model_id):
        if model_id in _LANGDEFAULTS:
            model_id = _LANGDEFAULTS.get(model_id)
        meta_path = f"{cls._XDG_PATH}/{model_id}.json"
        if isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            url = f"{cls._BASE_METADATA_URL}/{model_id}.json"
            meta = requests.get(url).json()
            with open(meta_path, "wb") as f:
                f.write(meta)

        model_path = f"{cls._XDG_PATH}/{model_id}.pkl"
        if not isfile(model_path):
            url = f"{cls._BASE_MODEL_URL}/{model_id}.pkl"
            model = requests.get(url).content
            with open(model_path, "wb") as f:
                f.write(model)

        return meta, model_path

    @classmethod
    def load_model(cls, model_id):
        from ovos_classifiers.tasks.tagger import OVOSBrillTagger, OVOSNgramTagger, OVOSClassifierTagger

        data, model_path = cls.get_model(model_id)
        if data["algo"] == "TrigramTagger":
            clf = OVOSNgramTagger.from_file(model_path)
        elif data["algo"] == "nltk.brill.fntbl37":
            clf = OVOSBrillTagger.from_file(model_path)
        else:
            clf = OVOSClassifierTagger.from_file(model_path)
        return data, clf

    def tag(self, utterances):
        if isinstance(utterances, str):
            utterances = [utterances]
        return self.clf.predict(utterances)


if __name__ == "__main__":
    sentences = [
        "The brown fox jumped over the lazy dog",
        "Turn off the TV",
        "Turn on the lights",
        "thats amazing",
        "what time is it",
        "tell me about einstein"
    ]
    p = OVOSUtteranceTagger("en")
    print(p.model_id)
    print(p.tagset)
    print(p.tag(sentences))
    # ['SENTENCE:STATEMENT' 'COMMAND:ACTION' 'COMMAND:ACTION'
    #  'SENTENCE:EXCLAMATION' 'QUESTION:QUERY' 'COMMAND:ACTION']

