from os.path import dirname, isfile
import json


class HeuristicUtteranceTagger:

    def __init__(self, config=None):
        self.config = config or {}
        lang = self.config.get("lang", "en-us")
        self.lang = lang.split("-")[0]
        self._get_kwords(self.lang)  # throw exception if lang unsupported

    def predict(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        return [self.classify(s, self.lang) for s in sentences]

    @staticmethod
    def _get_kwords(lang):
        res = f"{dirname(dirname(__file__))}/res/{lang}/utttags.json"
        if not isfile(res):
            raise ValueError(f"unsupported lang: {lang}")
        with open(res) as f:
            data = json.load(f)

        return data.get("command", []), \
               data.get("denial", []), \
               data.get("query", []), \
               data.get("request", []), \
               data.get("yesno", []), \
               data.get("exclamation", []), \
               data.get("social", [])

    @classmethod
    def classify(cls, sentence, lang):
        sentence = sentence.lower().strip()
        command_action_keywords, command_denial_keywords, question_query_keywords, \
        question_request_keywords, question_yesno_keywords, sentence_exclamation_keywords, \
        sentence_social_keywords = cls._get_kwords(lang)

        if any(sentence.startswith(w) for w in command_action_keywords):
            return "COMMAND:ACTION"
        elif any(sentence.startswith(w) for w in question_yesno_keywords):
            return "QUESTION:YESNO"
        elif any(sentence.startswith(w) for w in question_query_keywords):
            return "QUESTION:QUERY"
        elif any(w in sentence for w in command_denial_keywords):
            return "COMMAND:DENIAL"
        elif any(w in sentence for w in question_request_keywords):
            return "QUESTION:REQUEST"
        elif any(w in sentence for w in sentence_social_keywords):
            return "SENTENCE:SOCIAL"
        elif any(w in sentence for w in sentence_exclamation_keywords):
            return "SENTENCE:EXCLAMATION"
        else:
            return "SENTENCE:STATEMENT"


if __name__ == "__main__":
    sentence = "How much does it cost?"
    label = HeuristicUtteranceTagger.classify(sentence, "en")
    print(label)  # Output: "QUESTION:QUERY"

    sentence = "tem cerveja no frigorifico?"
    label = HeuristicUtteranceTagger.classify(sentence, "pt")
    print(label)  # Output: "QUESTION:YESNO"

    sentence = "quem inventou o telefone"
    label = HeuristicUtteranceTagger({"lang": "pt"}).predict(sentence)
    print(label)  # Output: "QUESTION:QUERY"
