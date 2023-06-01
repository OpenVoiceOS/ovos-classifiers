import json
from os import makedirs
from os.path import isfile

import requests
from ovos_config import Configuration
from ovos_utils.xdg_utils import xdg_data_home

from ovos_classifiers.heuristics.corefiob import CorefIOBTags, CorefIOBHeuristicTagger
from ovos_classifiers.utils import load_tagger

# TODO - benchmark and choose based on performance/model size
# TODO - ensure all langs have 1 model
_LANGDEFAULTS = {
    "en": "corefiob_heuristic",
    "pt": "corefiob_heuristic"
}


class OVOSCorefIOBTagger:
    _XDG_PATH = f"{xdg_data_home()}/OpenVoiceOS/classifiers"
    _BASE_METADATA_URL = "https://github.com/OpenVoiceOS/ovos-classifiers/raw/dev/models/metadata"
    _BASE_MODEL_URL = "https://github.com/OpenVoiceOS/ovos-classifiers/raw/dev/models/corefiob"
    makedirs(_XDG_PATH, exist_ok=True)

    def __init__(self, model_id=None):
        config_core = Configuration()
        self.config = config_core.get("classifiers", {}).get("corefiob", {})
        model_id = model_id or self.config.get("model_id") or "corefiob_heuristic"
        if model_id in _LANGDEFAULTS:
            model_id = _LANGDEFAULTS.get(model_id)
        self.model_id = model_id
        self.meta, self.clf = self.load_model(self.model_id)

    @property
    def tagset(self):
        return self.meta.get("tagset") or "CorefIOBTags"

    @classmethod
    def get_model(cls, model_id):
        if model_id in _LANGDEFAULTS:
            model_id = _LANGDEFAULTS.get(model_id)

        if model_id == "corefiob_heuristic":
            return {"model_id": "corefiob_heuristic",
                    "tagset": "CorefIOBTags",
                    "lang": Configuration().get("lang", "en-us"),
                    "algo": "heuristic"}, CorefIOBHeuristicTagger

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
        data, model_path = cls.get_model(model_id)
        return load_tagger(data, model_path)

    def iob_tag(self, postagged_tokens):
        return self.clf.tag(postagged_tokens)

    @staticmethod
    def normalize_corefs(iobtagged_tokens):
        return CorefIOBHeuristicTagger.normalize_corefs(iobtagged_tokens)
