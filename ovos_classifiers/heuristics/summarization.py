from ovos_classifiers.heuristics.tokenize import word_tokenize
from quebra_frases import sentence_tokenize
from heapq import nlargest
from nltk.corpus import stopwords
import nltk
from string import punctuation


class HeuristicSummarizer:
    def __init__(self):
        nltk.download("stopwords")
        self.langs = {
            "en": "english",
            "ar": "arabic",
            "az": "azerbaijani",
            "ca": "catalan",
            "eu": "basque",
            "da": "danish",
            "de": "german",
            "nl": "dutch",
            "fi": "finnish",
            "fr": "french",
            "hu": "hungarian",
            "it": "italian",
            "no": "norwegian",
            "pt": "portuguese",
            "ru": "russian",
            "es": "spanish",
            "sw": "swedish",
            "ro": "romanian"
        }

    def summarize(self, document, lang="en"):
        lang = lang.split("-")[0].lower()
        lang = self.langs.get(lang) or lang

        stop_words = stopwords.words(lang)

        tokens = word_tokenize(document, lang)

        word_frequencies = {}
        for word in tokens:
            if word.lower() not in stop_words:
                if word.lower() not in punctuation:
                    if word not in word_frequencies.keys():
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1

        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency

        sentence_scores = {}
        sent_token = document.split("\n")
        for sent in sent_token:
            sentence = sent.split(" ")
            for word in sentence:
                if word.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.lower()]

        select_length = int(len(sent_token) * 0.3)
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        final_summary = [word for word in summary]
        return '\n'.join(final_summary) or document


if __name__ == "__main__":
    doc = """
    Introducing OpenVoiceOS - The Free and Open-Source Personal Assistant and Smart Speaker.
    
    OpenVoiceOS is a new player in the smart speaker market, offering a powerful and flexible alternative to proprietary solutions like Amazon Echo and Google Home.
    
    With OpenVoiceOS, you have complete control over your personal data and the ability to customize and extend the functionality of your smart speaker.
    
    Built on open-source software, OpenVoiceOS is designed to provide users with a seamless and intuitive voice interface for controlling their smart home devices, playing music, setting reminders, and much more.
    
    The platform leverages cutting-edge technology, including machine learning and natural language processing, to deliver a highly responsive and accurate experience.
    
    In addition to its voice capabilities, OpenVoiceOS features a touch-screen GUI made using QT5 and the KF5 framework.
    
    The GUI provides an intuitive, user-friendly interface that allows you to access the full range of OpenVoiceOS features and functionality.
    
    Whether you prefer voice commands or a more traditional touch interface, OpenVoiceOS has you covered.
    
    One of the key advantages of OpenVoiceOS is its open-source nature, which means that anyone with the technical skills can contribute to the platform and help shape its future.
    
    Whether you're a software developer, data scientist, or just someone with a passion for technology, you can get involved and help build the next generation of personal assistants and smart speakers.
    
    With OpenVoiceOS, you have the option to run the platform fully offline, giving you complete control over your data and ensuring that your information is never shared with third parties. This makes OpenVoiceOS the perfect choice for anyone who values privacy and security.
    
    So if you're looking for a personal assistant and smart speaker that gives you the freedom and control you deserve, be sure to check out OpenVoiceOS today!
    """


    h = HeuristicSummarizer()
    print(h.summarize(doc, lang="en"))
    #     With OpenVoiceOS, you have the option to run the platform fully offline, giving you complete control over your data and ensuring that your information is never shared with third parties. This makes OpenVoiceOS the perfect choice for anyone who values privacy and security.
    #     Whether you're a software developer, data scientist, or just someone with a passion for technology, you can get involved and help build the next generation of personal assistants and smart speakers.
    #     Built on open-source software, OpenVoiceOS is designed to provide users with a seamless and intuitive voice interface for controlling their smart home devices, playing music, setting reminders, and much more.
    #     With OpenVoiceOS, you have complete control over your personal data and the ability to customize and extend the functionality of your smart speaker.
    #     So if you're looking for a personal assistant and smart speaker that gives you the freedom and control you deserve, be sure to check out OpenVoiceOS today!
    #     OpenVoiceOS is a new player in the smart speaker market, offering a powerful and flexible alternative to proprietary solutions like Amazon Echo and Google Home.
    #     One of the key advantages of OpenVoiceOS is its open-source nature, which means that anyone with the technical skills can contribute to the platform and help shape its future.
