class EnglishARPAHeuristicPhonemizer:
    subword_phonemes = {
        "E": "EH", "S": "S", "A": "AE", "O": "OW", "F": "F", "H": "HH", "T": "T", "L": "L", "I": "IH", "P": "P",
        "R": "R", "Y": "Y", "EG": "T", "SS": "Z", "N": "N", "D": "D", "C": "K", "ER": "Z", "M": "M", "GE": "ZH",
        "ME": "Z", "B": "B", "U": "AH", "TE": "Z", "W": "W", "UE": "UW", "IE": "IY", "G": "G", "NE": "Z", "SE": "Z",
        "AI": "EY", "CH": "CH", "X": "K S", "VE": "V", "BE": "UW", "EPE": "IH", "OI": "OY", "AY": "W", "WO": "W",
        "K": "K", "AL": "W", "FFI": "F", "V": "V", "Q": "K", "ES": "Z", "RE": "ZH", "WHA": "W", "YE": "Y", "EA": "IY",
        "ODE": "OW", "EW": "Y", "GG": "G", "CI": "SH", "YEE": "IY", "UNI": "Y", "WH": "W", "LLI": "L", "XC": "K",
        "HT": "T", "SSI": "SH", "XE": "K", "J": "JH", "OE": "OW", "EY": "T", "OUG": "UW", "QUI": "K", "YI": "Y",
        "TAT": "N", "NOI": "AH", "DD": "D", "SSA": "S", "ETT": "S", "ETE": "T", "HTY": "T", "SU": "ZH", "HNO": "N",
        "SI": "ZH", "CKA": "K", "ASO": "Z", "ZE": "Z", "IO": "Y", "TU": "UW", "ON": "Z", "Z": "Z", "WA": "W", "FA": "F",
        "SHI": "SH", "CCO": "K", "NN": "N", "OA": "OW", "ORA": "R", "MME": "M", "PP": "P", "NI": "UW", "ITI": "Z",
        "SIG": "IH", "RN": "ER", "NO": "N", "OP": "P", "ETA": "N", "TA": "T", "PA": "S", "NTE": "N", "SKE": "S",
        "INS": "IH", "KN": "N", "WR": "R", "TTO": "T", "JUD": "JH", "YO": "Y", "RY": "T", "IB": "AY", "APO": "P",
        "PEC": "P", "FI": "S", "JE": "JH", "FE": "IH", "RRE": "R", "HED": "EH", "IS": "Z", "VIE": "V", "PHO": "F",
        "OPA": "P", "ESM": "S", "EM": "TH", "OSS": "AA", "TY": "T", "HI": "Z", "LAU": "L", "GAI": "G", "DJ": "JH",
        "EB": "EH", "ISI": "Z", "RIN": "ER", "CQU": "K", "MO": "UW", "TIA": "SH", "OIC": "OY", "ALE": "V", "REA": "UH",
        "PT": "P", "SOC": "S", "SSO": "Z", "ORE": "ER", "MI": "M", "THO": "TH", "SHE": "SH", "SEA": "S", "SSE": "S",
        "HRY": "R", "EC": "EH", "IA": "Y", "WRO": "R", "RPS": "R", "GAT": "AH", "INN": "IH", "AYE": "R", "IVE": "T",
        "RPR": "P", "CID": "IH", "ERA": "R", "FT": "F", "GGE": "G", "ENG": "S", "ATM": "T", "UL": "AH", "ARE": "R",
        "OCU": "OW", "ASI": "Z", "CES": "S", "": "Z", "SCH": "SH", "EU": "Y UW", "ALI": "W", "SW": "S", "PPO": "P",
        "OZE": "OW", "HE": "HH", "BO": "B", "OOD": "UW", "WY": "OY", "OB": "OW", "SA": "Z", "CED": "S", "VA": "V",
        "AVE": "V", "XX": "K", "ERC": "ER", "BY": "B", "CHI": "SH", "UT": "Y", "ONU": "OW", "SSU": "SH", "GH": "G",
        "AIG": "EY", "OPE": "OW", "OOS": "UW", "EVE": "Z", "GI": "ZH", "OTH": "OW", "OTI": "OW", "MR": "M", "VI": "V",
        "UTH": "TH", "OLE": "OW", "DIE": "JH", "IAM": "AH", "HRI": "R", "LS": "L", "SY": "S", "TMA": "S", "HIC": "IH",
        "YRI": "IH", "MN": "M", "GUA": "G", "NY": "N", "AV": "V", "YBE": "B", "INT": "N", "NNA": "N", "NH": "N",
        "AD": "EY", "BI": "IH", "DG": "JH", "DU": "JH", "REI": "R", "OC": "OW", "ORF": "ER", "ESE": "Z", "LIA": "L",
        "HEI": "EH", "GO": "G", "UGU": "G", "OSA": "AH", "THU": "TH", "YU": "Y", "TES": "T", "AUT": "UW", "MMI": "M",
        "OVI": "AA", "UGH": "Y", "LON": "N", "YWE": "IY", "UTE": "Y", "CIE": "SH", "RAL": "ER", "HU": "Y", "CKE": "K",
        "ZZ": "Z", "MBE": "M", "PSY": "S AY", "OTE": "OW", "RD": "R", "LOR": "L", "TER": "T", "PHI": "F", "KK": "K",
        "VEL": "V", "ATT": "AE", "IRY": "R", "ANN": "AE", "IRA": "ER", "CY": "S", "YMP": "IH", "APP": "AE", "VEI": "V",
        "USE": "Z", "CZA": "Z", "NIS": "N", "NW": "N", "DT": "T", "MP": "M", "DJE": "JH", "DS": "D", "EVA": "AH",
        "RC": "S", "LD": "L", "HYS": "AH", "MY": "M", "OTT": "AA", "DDH": "D", "IU": "Y", "YST": "IH", "HRO": "R",
        "HAI": "AY", "NGU": "NG", "EGY": "IY", "OIS": "W", "ASA": "Z", "TTH": "TH", "ALA": "L", "OG": "OW", "RRO": "R",
        "OST": "OW", "IDG": "IH", "EAV": "EH", "DR": "D", "WI": "W", "TR": "T", "MPH": "M", "XI": "K", "NEA": "N",
        "GEO": "JH", "AAC": "Z", "DH": "D", "SIC": "Z", "VO": "V", "LES": "L", "IX": "IY", "HUG": "Y", "HOV": "OW",
        "KY": "K", "QU": "K", "SH": "SH", "TH": "TH", "DGE": "JH", "EE": "IY", "AO": "AW", "CK": "K", "TT": "T",
        "EI": "IY", "GHT": "T", "PH": "F", "GN": "N", "FF": "F", "OO": "UW", "UA": "W", "NG": "NG", "BB": "B",
        "TCH": "CH", "ARA": "R", "OWE": "OW", "KH": "K", "ZA": "Z", "RR": "R", "CC": "K S", "IGH": "AY", "EIGH": "EY"}
    avg_durs = {
        '.': 2.605,
        'AA': 2.559,
        'AE': 2.598,
        'AH': 2.793,
        'AO': 2.753,
        'AW': 2.877,
        'AX': 2.746,
        'AY': 2.845,
        'B': 2.525,
        'CH': 2.851,
        'D': 2.861,
        'DH': 2.445,
        'EH': 2.363,
        'ER': 2.799,
        'EY': 2.811,
        'F': 2.732,
        'G': 2.54,
        'HH': 2.571,
        'IH': 2.728,
        'IY': 2.762,
        'JH': 2.448,
        'K': 2.675,
        'L': 2.651,
        'M': 2.642,
        'N': 2.823,
        'NG': 2.828,
        'OW': 2.713,
        'OY': 2.768,
        'P': 2.661,
        'R': 2.815,
        'S': 2.618,
        'SH': 2.755,
        'T': 2.857,
        'TH': 3.239,
        'UH': 2.596,
        'UW': 2.804,
        'V': 2.847,
        'W': 2.711,
        'Y': 2.633,
        'Z': 2.91,
        'ZH': 3.04}

    @classmethod
    def subword_tokenize(cls, sentence):
        if not sentence:
            return []
        words = sentence.split()
        subwords = []
        for word in words:
            curr = word.upper()
            while curr:
                w3 = curr[:3]
                w2 = curr[:2]
                w1 = curr[0]
                if w3 in cls.subword_phonemes:
                    subwords.append(w3)
                    curr = curr[3:]
                elif w2 in cls.subword_phonemes:
                    subwords.append(w2)
                    curr = curr[2:]
                elif w1 in cls.subword_phonemes:
                    subwords.append(w1)
                    curr = curr[1:]
                else:  # what do?
                    curr = curr[1:]
            subwords.append(".")
        subwords = subwords[:-1]  # drop last "."
        return subwords

    @classmethod
    def phonemize(cls, sentence):
        subwords = cls.subword_tokenize(sentence)
        return [cls.subword_phonemes.get(s, ".") for s in subwords]

    @classmethod
    def phoneme_duration_tokenize(cls, sentence):
        subwords = cls.subword_tokenize(sentence)
        return [(cls.subword_phonemes.get(s, "."), cls.avg_durs.get(s, 3.1)) for s in subwords]
