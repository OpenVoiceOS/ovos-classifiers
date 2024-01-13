import math
from ovos_classifiers.heuristics.tokenize import word_tokenize


class BM25:
    """
    Best Match 25. - taken from http://ethen8181.github.io/machine-learning/search/bm25_intro.html

    Parameters
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.

    doc_len_ : list[int]
        Number of terms per document. So [3] means the first
        document contains 3 terms.

    corpus_ : list[list[str]]
        The input corpus.

    corpus_size_ : int
        Number of documents in the corpus.

    avg_doc_len_ : float
        Average number of terms for documents in the corpus.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1

    def fit(self, corpus):
        """
        Fit the various statistics that are required to calculate BM25 ranking
        score using the corpus given.

        Parameters
        ----------
        corpus : list[list[str]]
            Each element in the list represents a document, and each document
            is a list of the terms.

        Returns
        -------
        self
        """
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            tf.append(frequencies)

            # compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def search(self, query):
        scores = [self._score(query, index) for index in range(self.corpus_size_)]
        return scores

    def _score(self, query, index):
        score = 0.0

        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)

        return score


def rank_answers(question, evidence, stopwords=None):
    stopwords = stopwords or []
    bm25 = BM25()
    corpus = [[w.lower() for w in word_tokenize(s) if w.lower() not in stopwords]
              for s in evidence]
    bm25.fit(corpus)
    scores = bm25.search(word_tokenize(question.lower()))
    return {k: v for (k, v) in zip(evidence, scores)}


def get_best_answer(question, evidence, stopwords=None):
    scores = rank_answers(question, evidence, stopwords)
    best = max(scores, key=scores.get)
    ties = [t for t, s in scores.items() if s >= scores[best]]
    return min(ties, key=len)


if __name__ == "__main__":
    from ovos_classifiers.utils import get_stopwords

    stopwords = get_stopwords("en")

    q = "who is Einstein"
    ans = [
        "Albert Einstein was a German–born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.",
        "The Einstein family is the family of physicist Albert Einstein (1879–1955)."
    ]

    print(rank_answers(q, ans, stopwords))
    #  {'Albert Einstein was a German–born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.': 0.17298060416883737,
    #  'The Einstein family is the family of physicist Albert Einstein (1879–1955).': 0.27090870251702015}
    print(get_best_answer(q, ans, stopwords))
    # The Einstein family is the family of physicist Albert Einstein (1879–1955).

    q = "what is the speed of light"
    ans = [
        "The speed of light in vacuum, commonly denoted c, is a universal physical constant that is exactly equal to 299,792,458 metres per second (approximately 300,000 kilometres per second; 186,000 miles per second; 671 million miles per hour).",
        "The speed of light in vacuum, commonly denoted c, is a universal physical constant that is exactly equal to 299,792,458 metres per second.",
        "The speed of light has a value of about 300 million meters per second"
    ]

    print(rank_answers(q, ans, stopwords))
    # {'The speed of light in vacuum, commonly denoted c, is a universal physical constant that is exactly equal to 299,792,458 metres per second (approximately 300,000 kilometres per second; 186,000 miles per second; 671 million miles per hour).': 0.19742903205224116,
    # 'The speed of light in vacuum, commonly denoted c, is a universal physical constant that is exactly equal to 299,792,458 metres per second.': 0.28071940494928044,
    # 'The speed of light has a value of about 300 million meters per second': 0.3837407011345503}
    print(get_best_answer(q, ans, stopwords))
    # The speed of light has a value of about 300 million meters per second

    q = "who is Isaac Newton"
    ans = [
        "Sir Isaac Newton was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, and author who was described in his time as a natural philosopher.",
        "Sir Isaac Newton (25 December 1642 – 20 March 1726/27) was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, and author who was described in his time as a natural philosopher."
    ]

    print(rank_answers(q, ans, stopwords))
    # {'Sir Isaac Newton was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, and author who was described in his time as a natural philosopher.': 0.3977924875504463,
    # 'Sir Isaac Newton (25 December 1642 – 20 March 1726/27) was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, and author who was described in his time as a natural philosopher.': 0.3365936433119162}
    print(get_best_answer(q, ans, stopwords))
    # Sir Isaac Newton was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, and author who was described in his time as a natural philosopher.

    q = "who invented the telephone"
    ans = [
        "A telephone is a telecommunications device that permits two or more users to conduct a conversation when they are too far apart to be easily heard directly.",
        "The telephone was invented by Alexander Graham Bell and Antonio Meucci"
    ]

    print(rank_answers(q, ans, stopwords))
    # {'A telephone is a telecommunications device that permits two or more users to conduct a conversation when they are too far apart to be easily heard directly.': 0.15854048416865615,
    # 'The telephone was invented by Alexander Graham Bell and Antonio Meucci': 1.0299632204163527}
    print(get_best_answer(q, ans, stopwords))
    # The telephone was invented by Alexander Graham Bell and Antonio Meucci

    q = "who invented the internet"
    ans = [
        "Al Gore is a United States politician who served successively in the House of Representatives, the Senate, and as the Vice President from 1993 to 2001.",
        "The answer is Advanced Research Projects Agency (ARPA), Tom Truscott, Jim Ellis, and 4 more"
    ]

    print(rank_answers(q, ans, stopwords))
    #  {'Al Gore is a United States politician who served successively in the House of Representatives, the Senate, and as the Vice President from 1993 to 2001.': 0.0,
    #  'The answer is Advanced Research Projects Agency (ARPA), Tom Truscott, Jim Ellis, and 4 more': 0.0}
    print(get_best_answer(q, ans, stopwords))
    # The answer is Advanced Research Projects Agency (ARPA), Tom Truscott, Jim Ellis, and 4 more

    q = "when will the world end"
    ans = [
        "The world will effectively end 5 billion years from now when the Sun becomes a red giant star",
        "Predictions of apocalyptic events that would result in the extinction of humanity, a collapse of civilization, or the destruction of the planet have been made since at least the beginning of the Common Era."
    ]

    print(rank_answers(q, ans, stopwords))
    # {'The world will effectively end 5 billion years from now when the Sun becomes a red giant star': 1.5946243114922676,
    # 'Predictions of apocalyptic events that would result in the extinction of humanity, a collapse of civilization, or the destruction of the planet have been made since at least the beginning of the Common Era.': 0.0}
    print(get_best_answer(q, ans, stopwords))
    # The world will effectively end 5 billion years from now when the Sun becomes a red giant star
