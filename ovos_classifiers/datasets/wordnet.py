import nltk
from nltk.corpus import wordnet as wn


class Wordnet:
    nltk.download("wordnet")
    nltk.download('omw-1.4')

    @staticmethod
    def get_synsets(word, pos=wn.NOUN):
        synsets = wn.synsets(word, pos=pos)
        if not len(synsets):
            return []
        return synsets

    @staticmethod
    def get_definition(word, pos=wn.NOUN, synset=None):
        if synset is None:
            synsets = wn.synsets(word, pos=pos)
            if not len(synsets):
                return []
            synset = synsets[0]
        return synset.definition()

    @staticmethod
    def get_examples(word, pos=wn.NOUN, synset=None):
        if synset is None:
            synsets = wn.synsets(word, pos=pos)
            if not len(synsets):
                return []
            synset = synsets[0]
        return synset.examples()

    @staticmethod
    def get_lemmas(word, pos=wn.NOUN, synset=None):
        if synset is None:
            synsets = wn.synsets(word, pos=pos)
            if not len(synsets):
                return []
            synset = synsets[0]
        return [l.name().replace("_", " ") for l in synset.lemmas()]

    @staticmethod
    def get_hypernyms(word, pos=wn.NOUN, synset=None):
        if synset is None:
            synsets = wn.synsets(word, pos=pos)
            if not len(synsets):
                return []
            synset = synsets[0]
        return [l.name().split(".")[0].replace("_", " ") for l in
                synset.hypernyms()]

    @staticmethod
    def get_hyponyms(word, pos=wn.NOUN, synset=None):
        if synset is None:
            synsets = wn.synsets(word, pos=pos)
            if not len(synsets):
                return []
            synset = synsets[0]
        return [l.name().split(".")[0].replace("_", " ") for l in
                synset.hyponyms()]

    @staticmethod
    def get_holonyms(word, pos=wn.NOUN, synset=None):
        if synset is None:
            synsets = wn.synsets(word, pos=pos)
            if not len(synsets):
                return []
            synset = synsets[0]
        return [l.name().split(".")[0].replace("_", " ") for l in
                synset.member_holonyms()]

    @staticmethod
    def get_root_hypernyms(word, pos=wn.NOUN, synset=None):
        if synset is None:
            synsets = wn.synsets(word, pos=pos)
            if not len(synsets):
                return []
            synset = synsets[0]
        return [l.name().split(".")[0].replace("_", " ") for l in
                synset.root_hypernyms()]

    @staticmethod
    def common_hypernyms(word, word2, pos=wn.NOUN):
        synsets = wn.synsets(word, pos=pos)
        if not len(synsets):
            return []
        synset = synsets[0]
        synsets = wn.synsets(word2, pos=pos)
        if not len(synsets):
            return []
        synset2 = synsets[0]
        return [l.name().split(".")[0].replace("_", " ") for l in
                synset.lowest_common_hypernyms(synset2)]

    @staticmethod
    def get_antonyms(word, pos=wn.NOUN, synset=None):
        if synset is None:
            synsets = wn.synsets(word, pos=pos)
            if not len(synsets):
                return []
            synset = synsets[0]
        lemmas = synset.lemmas()
        if not len(lemmas):
            return []
        lemma = lemmas[0]
        antonyms = lemma.antonyms()
        return [l.name().split(".")[0].replace("_", " ") for l in antonyms]

    @classmethod
    def query(cls, query, pos=wn.NOUN, synset=None):
        if synset is None:
            synsets = wn.synsets(query, pos=pos)
            if not len(synsets):
                return {}
            synset = synsets[0]
        res = {"lemmas": cls.get_lemmas(query, pos=pos, synset=synset),
               "antonyms": cls.get_antonyms(query, pos=pos, synset=synset),
               "holonyms": cls.get_holonyms(query, pos=pos, synset=synset),
               "hyponyms": cls.get_hyponyms(query, pos=pos, synset=synset),
               "hypernyms": cls.get_hypernyms(query, pos=pos, synset=synset),
               "root_hypernyms": cls.get_root_hypernyms(query, pos=pos, synset=synset),
               "definition": cls.get_definition(query, pos=pos, synset=synset)}
        return res
