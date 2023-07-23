import re


class EnglishARPAHeuristicPhonemizer:
    @staticmethod
    def phonemize(word):
        word = re.sub('[^a-zA-Z0-9 \n\.]', " ", word.lower())
        total_phones = []
        for w in word.split(" "):

            basic_pronunciations = {'a': ['AE'], 'b': ['B'], 'c': ['K'],
                                    'd': ['D'],
                                    'e': ['EH'], 'f': ['F'], 'g': ['G'],
                                    'h': ['HH'],
                                    'i': ['IH'],
                                    'j': ['JH'], 'k': ['K'], 'l': ['L'],
                                    'm': ['M'],
                                    'n': ['N'], 'o': ['OW'], 'p': ['P'],
                                    'qu': ['K', 'W'], 'r': ['R'],
                                    's': ['S'], 't': ['T'], 'u': ['AH'],
                                    'v': ['V'],
                                    'w': ['W'], 'x': ['K', 'S'], 'y': ['Y'],
                                    'z': ['Z'], 'ch': ['CH'],
                                    'sh': ['SH'], 'th': ['TH'], 'dg': ['JH'],
                                    'dge': ['JH'], 'psy': ['S', 'AY'],
                                    'oi': ['OY'],
                                    'ee': ['IY'],
                                    'ao': ['AW'], 'ck': ['K'], 'tt': ['T'],
                                    'nn': ['N'], 'ai': ['EY'], 'eu': ['Y', 'UW'],
                                    'ue': ['UW'],
                                    'ie': ['IY'], 'ei': ['IY'], 'ea': ['IY'],
                                    'ght': ['T'], 'ph': ['F'], 'gn': ['N'],
                                    'kn': ['N'], 'wh': ['W'],
                                    'wr': ['R'], 'gg': ['G'], 'ff': ['F'],
                                    'oo': ['UW'], 'ua': ['W', 'AO'], 'ng': ['NG'],
                                    'bb': ['B'],
                                    'tch': ['CH'], 'rr': ['R'], 'dd': ['D'],
                                    'cc': ['K', 'S'], 'oe': ['OW'],
                                    'igh': ['AY'], 'eigh': ['EY']}
            phones = []
            progress = len(w) - 1
            while progress >= 0:
                if w[0:3] in basic_pronunciations.keys():
                    for phone in basic_pronunciations[w[0:3]]:
                        phones.append(phone)
                    w = w[3:]
                    progress -= 3
                elif w[0:2] in basic_pronunciations.keys():
                    for phone in basic_pronunciations[w[0:2]]:
                        phones.append(phone)
                    w = w[2:]
                    progress -= 2
                elif w[0] in basic_pronunciations.keys():
                    for phone in basic_pronunciations[w[0]]:
                        phones.append(phone)
                    w = w[1:]
                    progress -= 1
                else:
                    # skip this word letter by letter, if we can't handle it.
                    w = w[1:]
                    progress -= 1
            if len(total_phones):
                total_phones.append(".")
            total_phones += phones
        return total_phones
