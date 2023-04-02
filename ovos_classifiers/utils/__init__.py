import re

import nltk
from nltk.stem.snowball import SnowballStemmer


def load_tagger(data, model_path):
    from ovos_classifiers.tasks.tagger import OVOSBrillTagger, OVOSNgramTagger

    if data["algo"] == "heuristic":
        clazz = model_path
        clf = clazz(data)
    elif data["algo"] == "TrigramTagger":
        clf = OVOSNgramTagger.from_file(model_path)
    elif data["algo"] == "nltk.brill.fntbl37":
        clf = OVOSBrillTagger.from_file(model_path)
    elif data["algo"] == "sklearn.ensemble.VotingClassifier":
        from ovos_classifiers.skovos.tagger import SklearnOVOSClassifierTagger
        clf = SklearnOVOSClassifierTagger.from_file(model_path)
    elif "sklearn." in data["algo"]:
        from ovos_classifiers.skovos.tagger import SklearnOVOSVotingClassifierTagger
        clf = SklearnOVOSVotingClassifierTagger.from_file(model_path)
    else:
        raise ValueError(f"unknown model format: {data['algo']}")
    return data, clf


def load_classifier(data, model_path):
    if data["algo"] == "heuristic":
        try:
            clazz = model_path
            clf = clazz(data)
        except:
            clf = None
    elif data["algo"] == "sklearn.ensemble.VotingClassifier":
        from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier
        clf = SklearnOVOSClassifier.from_file(model_path)
    elif "sklearn." in data["algo"]:
        from ovos_classifiers.skovos.classifier import SklearnOVOSVotingClassifier
        clf = SklearnOVOSVotingClassifier.from_file(model_path)
    else:
        raise ValueError(f"unknown model format: {data['algo']}")
    return data, clf


def get_stemmer(lang="porter"):
    languages = {
        "ar": "arabic",
        "da": "danish",
        "nl": "dutch",
        "en": "english",
        "fi": "finnish",
        "fr": "french",
        "de": "german",
        "hu": "hungarian",
        "it": "italian",
        "no": "norwegian",
        "pt": "portuguese",
        "ro": "romanian",
        "ru": "russian",
        "es": "spanish",
        "sw": "swedish",
    }
    if lang == "dummy":
        return DummyStemmer()
    lang = lang.split("-")[0]
    if lang in languages:
        return SnowballStemmer(languages[lang])
    else:
        return SnowballStemmer('porter')


def get_word_shape(word):
    word_shape = 'other'
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        word_shape = 'number'
    elif re.match('\W+$', word):
        word_shape = 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        word_shape = 'capitalized'
    elif re.match('[A-Z]+$', word):
        word_shape = 'uppercase'
    elif re.match('[a-z]+$', word):
        word_shape = 'lowercase'
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        word_shape = 'camelcase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        word_shape = 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        word_shape = 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        word_shape = 'contains-hyphen'

    return word_shape


def extract_iob_features(tokens, index, history, stemmer=None, memory=2):
    """
    `tokens`  = a POS-tagged postagged_tokens [(w1, t1), ...]
    `index`   = the index of the token we want to extract utils for
    `history` = the previous predicted IOB tags
    """
    feat_dict = extract_postag_features(tokens, index, stemmer=stemmer,
                                        memory=memory)

    # Pad the sequence with placeholders
    tokens = ['O'] * memory + history

    index += memory

    # look back N predictions
    for i in range(1, memory + 1):
        k = "prev-" * i
        previob = tokens[index - i]
        # update with IOB utils
        feat_dict[k + "iob"] = previob

    return feat_dict


def extract_postag_features(tokens, index, stemmer=None, memory=2):
    """
    `tokens`  = a POS-tagged postagged_tokens [(w1, t1), ...]
    `index`   = the index of the token we want to extract utils for
    """
    original_toks = list(tokens)
    # word utils
    feat_dict = extract_word_features([t[0] for t in tokens],
                                      index, stemmer, memory=memory)

    # Pad the sequence with placeholders
    tokens = []
    for i in range(1, memory + 1):
        tokens.append((f'__START{i}__', f'__START{i}__'))
    tokens = list(reversed(tokens)) + original_toks
    for i in range(1, memory + 1):
        tokens.append((f'__END{i}__', f'__END{i}__'))

    # shift the index to accommodate the padding
    index += memory

    word, pos = tokens[index]

    # update with postag utils
    feat_dict["pos"] = pos

    # look ahead N words
    for i in range(1, memory + 1):
        k = "next-" * i
        nextword, nextpos = tokens[index + i]
        feat_dict[k + "pos"] = nextpos

    # look back N words
    for i in range(1, memory + 1):
        k = "prev-" * i
        prevword, prevpos = tokens[index - i]
        feat_dict[k + "pos"] = prevpos

    return feat_dict


def extract_word_features(tokens, index=0, stemmer=None, memory=2):
    """
    `tokens`  = a tokenized postagged_tokens [w1, w2, ...]
    `index`   = the index of the token we want to extract utils for
    """
    stemmer = stemmer or get_stemmer()
    if isinstance(tokens, str):
        tokens = [tokens]
    original_toks = list(tokens)

    # Pad the sequence with placeholders
    tokens = []
    for i in range(1, memory + 1):
        tokens.append(f'__START{i}__')
    tokens = list(reversed(tokens)) + original_toks
    for i in range(1, memory + 1):
        tokens.append(f'__END{i}__')

    # shift the index to accommodate the padding
    index += memory

    word = tokens[index]
    feat_dict = extract_single_word_features(word)
    feat_dict["word"] = word
    feat_dict["shape"] = get_word_shape(word)
    feat_dict["lemma"] = stemmer.stem(word)

    # look ahead N words
    for i in range(1, memory + 1):
        k = "next-" * i
        nextword = tokens[index + i]
        feat_dict[k + "word"] = nextword
        feat_dict[k + "lemma"] = stemmer.stem(nextword)
        feat_dict[k + "shape"] = get_word_shape(nextword)

    # look back N words
    for i in range(1, memory + 1):
        k = "prev-" * i
        prevword = tokens[index - i]
        feat_dict[k + "word"] = prevword
        feat_dict[k + "lemma"] = stemmer.stem(prevword)
        feat_dict[k + "shape"] = get_word_shape(prevword)

    return feat_dict


def extract_single_word_features(word, lowercase=True):
    if lowercase:
        word = word.lower()
    feat_dict = {
        'suffix1': word[-1:],
        'suffix2': word[-2:],
        'suffix3': word[-3:],
        'prefix1': word[:1],
        'prefix2': word[:2],
        'prefix3': word[:3]
    }
    return feat_dict


def extract_rte_features(rtepair):
    extractor = nltk.RTEFeatureExtractor(rtepair)
    features = {}
    features['word_overlap'] = len(extractor.overlap('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
    features['ne_overlap'] = len(extractor.overlap('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
    return features


def normalize(X, stemmer=None, lemmatize=True):
    documents = []

    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        if lemmatize:
            stemmer = stemmer or get_stemmer()
            document = document.split()
            try:
                document = [stemmer.lemmatize(word) for word in document]
            except:
                pass

            document = ' '.join(document)

        documents.append(document)
    return documents


class DummyStemmer:
    def stem(self, word):
        return word.rstrip("s")

    def lemmatize(self, word):
        return self.stem(word)
