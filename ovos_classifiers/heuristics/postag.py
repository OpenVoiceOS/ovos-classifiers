import re

import nltk
from ovos_classifiers.heuristics.tokenize import word_tokenize


class RegexPostag:
    def __init__(self, config=None):
        self.config = config or {}
        self.lang = self.config.get("lang", "en-us")

    def tag(self, sentence):
        if isinstance(sentence, str):
            sentence = word_tokenize(sentence, lang=self.lang)
        if self.lang.startswith("en"):
            return self.tag_en(sentence)
        elif self.lang.startswith("es"):
            return self.tag_es(sentence)
        elif self.lang.startswith("pt"):
            return self.tag_pt(sentence)
        raise ValueError(f"unsupported lang: {self.lang}")

    # lang specific regexes
    def tag_en(self, sentence):
        if isinstance(sentence, str):
            sentence = word_tokenize(sentence, lang="en")
        tags = []
        for token in sentence:
            if re.match(r'^([Tt]his|[Tt]hat|[Aa]|[Tt]he|[Aa]ny)$', token):
                tags.append('DET')
            elif re.match(r'^([Ii]|[Mm]e|[Yy]ou|[Hh]e|[Ss]he|[Ii]t|[Ww]e|[Tt]hey)(\'s)$', token):
                tags.append('PRON')
            elif re.match(r'^([Hh]ave|[Hh]as|[Hh]ad|[Cc]an|[Mm]ay|[Ss]hall|[Ww]ill|[Ss]hould|[Mm]ust)$', token):
                tags.append('AUX')
            elif re.match(r'^([Vv]ery|[Rr]ather|[Qq]uite|[Tt]oo|[Nn]ot|[Nn]ever|[Aa]lways|[Ss]eldom|[Oo]ften)$', token):
                tags.append('ADV')
            elif re.match(r'^([Ii]|[Mm]e|[Yy]ou|[Hh]im|[Hh]er|[Ii]t|[Uu]s|[Tt]hem)$', token):
                tags.append('PRON')
            elif re.match(r'^([Cc]an|[Ww]ill|[Mm]ay|[Ss]hould|[Mm]ust)$', token):
                tags.append('VERB')
            elif re.match(r'^([Aa]fter|[Bb]efore|[Ww]hile|[Ss]ince|[Uu]ntil|[Oo]f|[Ii]n|[Aa]t|[Oo]n|[Tt]o)$', token):
                tags.append('ADP')
            elif re.match(r'^[A-Z][a-z]*$', token):
                tags.append('PROPN')
            elif re.match(r'^\W+$', token):
                tags.append('PUNCT')
            elif re.match(r'^[a-z]*ly$', token):
                tags.append('ADV')
            elif re.match(r'^\d+(\.\d+)?$', token):
                tags.append('NUM')
            elif re.match(r'^[a-z]+(ed|ing|s)$', token):
                tags.append('VERB')
            else:
                tags.append('NOUN')
        return list(zip(sentence, tags))

    def tag_pt(self, sentence):
        if isinstance(sentence, str):
            sentence = word_tokenize(sentence, lang="pt")

        tags = []
        for token in sentence:
            # Determiners
            if re.match(r'^[oa]s?$', token, re.IGNORECASE):
                tags.append('DET')
            # Pronouns
            elif re.match(r'^(eu|tu|ele|ela|n[oã]s|v[oô]s|eles|elas)$', token, re.IGNORECASE):
                tags.append('PRON')
            # Verbs
            elif re.match(r'^\w+(ar|er|ir)$', token):
                tags.append('VERB')
            # Adverbs
            elif re.match(r'^\w+mente$', token):
                tags.append('ADV')
            # Punctuation
            elif re.match(r'^[,.:;!?()]$', token):
                tags.append('PUNCT')
            else:
                tags.append('NOUN')

        return list(zip(sentence, tags))

    def tag_es(self, sentence):
        if isinstance(sentence, str):
            sentence = word_tokenize(sentence, lang="es")

        tagged_tokens = []
        for token in sentence:
            # Check for punctuation
            if re.match(r'[^\w\s]+', token):
                tagged_tokens.append((token, 'PUNCT'))
            # Check for numbers
            elif re.match(r'\d+', token):
                tagged_tokens.append((token, 'NUM'))
            # Check for pronouns
            elif token.lower() in ['yo', 'tú', 'él', 'ella', 'usted', 'nosotros', 'nosotras', 'vosotros', 'vosotras',
                                   'ellos', 'ellas', 'ustedes']:
                tagged_tokens.append((token, 'PRON'))
            # Check for verbs
            elif re.match(r'(a|e|i|o|u)[a-z]*(ar|er|ir)', token.lower()):
                tagged_tokens.append((token, 'VERB'))
            # Check for adjectives
            elif re.match(r'[a-z]+[o|a|os|as]$', token.lower()):
                tagged_tokens.append((token, 'ADJ'))
            # Check for adverbs
            elif re.match(r'[a-z]+mente$', token.lower()):
                tagged_tokens.append((token, 'ADV'))
            # Check for determiners
            elif token.lower() in ['el', 'la', 'los', 'las', 'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos',
                                   'esas', 'aquel', 'aquella', 'aquellos', 'aquellas', 'un', 'una', 'unos', 'unas',
                                   'mi', 'tu', 'su', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'este', 'esta',
                                   'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 'aquel', 'aquella', 'aquellos',
                                   'aquellas']:
                tagged_tokens.append((token, 'DET'))
            # Check for prepositions
            elif token.lower() in ['a', 'ante', 'bajo', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre',
                                   'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin', 'sobre', 'tras']:
                tagged_tokens.append((token, 'ADP'))
            # Check for conjunctions
            elif token.lower() in ['y', 'e', 'ni', 'o', 'u', 'o bien', 'ya sea', 'ya', 'aunque', 'si', 'pero', 'sino',
                                   'como', 'porque', 'pues', 'entonces', 'luego', 'así que', 'por consiguiente',
                                   'mientras', 'cuando', 'antes', 'después', 'como', 'tal como', 'tan como']:
                tagged_tokens.append((token, 'CONJ'))
            # Check for interjections
            elif re.match(r'¡+|\!+', token):
                tagged_tokens.append((token, 'INTJ'))

            elif re.match(r'^[A-Z][a-záéíóúñü]*$', token):
                tagged_tokens.append((token, 'PROPN'))
            elif re.match(r'^[a-záéíóúñü]+$', token):
                tagged_tokens.append((token, 'NOUN'))
            elif re.match(r'^[a-záéíóúñü]+mente$', token):
                tagged_tokens.append((token, 'ADV'))
            elif re.match(r'^[a-záéíóúñü]+(ar|er|ir)$', token):
                tagged_tokens.append((token, 'VERB'))
            # If none of the above, assume it's a noun
            else:
                tagged_tokens.append((token, 'NOUN'))
        return tagged_tokens


class NltkPostag:
    def __init__(self, config=None):
        # TODO - lang support
        self.config = config or {}

    def tag(self, sentence):
        if isinstance(sentence, str):
            sentence = nltk.word_tokenize(sentence)
        return nltk.pos_tag(sentence, tagset="universal")
