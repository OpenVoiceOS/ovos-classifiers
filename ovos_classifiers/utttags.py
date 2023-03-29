import json
from os import makedirs
from os.path import isfile

import requests
from ovos_config import Configuration
from ovos_utils.xdg_utils import xdg_data_home

from ovos_classifiers.heuristics.utttags import HeuristicUtteranceTagger
from ovos_classifiers.utils import load_classifier

# TODO - benchmark and choose based on performance/model size
# TODO - ensure all langs have 1 model
_LANGDEFAULTS = {
    "en": "heuristic",
    "es": "heuristic",
    "pt": "heuristic",
    "uk": "heuristic",
    "de": "heuristic",
    "nl": "heuristic",
    "fr": "heuristic",
    "it": "heuristic",
    "sv": "heuristic",
    "no": "heuristic"

}


class OVOSUtteranceTagger:
    _XDG_PATH = f"{xdg_data_home()}/OpenVoiceOS/classifiers"
    _BASE_METADATA_URL = "https://github.com/OpenVoiceOS/ovos-classifiers/raw/dev/models/metadata"
    _BASE_MODEL_URL = "https://github.com/OpenVoiceOS/ovos-classifiers/raw/dev/models/utttags"
    makedirs(_XDG_PATH, exist_ok=True)

    def __init__(self, model_id=None):
        config_core = Configuration()
        self.lang = config_core.get("lang", "en-us")
        self.config = config_core.get("classifiers", {}).get("utttags", {})
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

        if model_id == "heuristic":
            return {"model_id": "heuristic",
                    "tagset": "utttags",
                    "lang": self.lang,
                    "algo": "heuristic"}, HeuristicUtteranceTagger

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
        data, model_path = self.get_model(model_id)
        return load_classifier(data, model_path)

    def predict(self, utterances):
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
        "can birds fly",
        "tell me about einstein"
    ]
    p = OVOSUtteranceTagger("en")
    print(p.model_id)
    print(p.tagset)
    print(p.predict(sentences))
    # ['SENTENCE:STATEMENT' 'COMMAND:ACTION' 'COMMAND:ACTION'
    #  'SENTENCE:EXCLAMATION' 'QUESTION:QUERY' 'COMMAND:ACTION']
