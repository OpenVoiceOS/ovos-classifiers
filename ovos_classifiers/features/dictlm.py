# it's a basic language model inside of a python dict

import random
from collections import defaultdict

from nltk import trigrams


class DictLM:
    def __init__(self):
        self.lm = defaultdict(lambda: defaultdict(lambda: 0))

    def train(self, sentences):
        for sentence in sentences:
            for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
                self.lm[(w1, w2)][w3] += 1

        # transform the counts to probabilities
        for w1_w2 in self.lm:
            total_count = float(sum(self.lm[w1_w2].values()))
            for w3 in self.lm[w1_w2]:
                self.lm[w1_w2][w3] /= total_count

    def generate(self):
        text = [None, None]
        prob = 1.0

        sentence_finished = False

        while not sentence_finished:
            r = random.random()
            accumulator = .0

            for word in self.lm[tuple(text[-2:])].keys():
                accumulator += self.lm[tuple(text[-2:])][word]

                if accumulator >= r:
                    # Update the probability with the conditional probability of the new word
                    prob *= self.lm[tuple(text[-2:])][word]
                    text.append(word)
                    break

            if text[-2:] == [None, None]:
                sentence_finished = True

        text = ' '.join([t for t in text if t])
        return text, prob


if __name__ == "__main__":
    import nltk
    from nltk.corpus import reuters

    nltk.download('reuters')
    lm = DictLM()
    lm.train(reuters.sents())
    for i in range(20):
        print(lm.generate())
