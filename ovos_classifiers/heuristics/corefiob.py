import enum
import json
from os.path import isfile, dirname


# different langs may use different subsets only
# eg, portuguese does not have inanimate or neutral
#     english does not have plural_(fe)male
class CorefIOBTags(str, enum.Enum):
    COREF_MALE = "B-COREF-MALE"
    COREF_FEMALE = "B-COREF-FEMALE"
    COREF_PLURAL = "B-COREF-PLURAL"
    COREF_PLURAL_MALE = "B-COREF-PLURAL-MALE"
    COREF_PLURAL_FEMALE = "B-COREF-PLURAL-FEMALE"
    COREF_NEUTRAL = "B-COREF-NEUTRAL"
    COREF_INANIMATE = "B-COREF-INANIMATE"

    ENTITY_MALE = "B-ENTITY-MALE"
    ENTITY_FEMALE = "B-ENTITY-FEMALE"
    ENTITY_PLURAL = "B-ENTITY-PLURAL"
    ENTITY_PLURAL_MALE = "B-ENTITY-PLURAL-MALE"
    ENTITY_PLURAL_FEMALE = "B-ENTITY-PLURAL-FEMALE"
    ENTITY_NEUTRAL = "B-ENTITY-NEUTRAL"
    ENTITY_INANIMATE = "B-ENTITY-INANIMATE"

    ENTITY_MALE_I = "I-ENTITY-MALE"
    ENTITY_FEMALE_I = "I-ENTITY-FEMALE"
    ENTITY_PLURAL_I = "I-ENTITY-PLURAL"
    ENTITY_PLURAL_MALE_I = "I-ENTITY-PLURAL-MALE"
    ENTITY_PLURAL_FEMALE_I = "I-ENTITY-PLURAL-FEMALE"
    ENTITY_NEUTRAL_I = "I-ENTITY-NEUTRAL"
    ENTITY_INANIMATE_I = "I-ENTITY-INANIMATE"


class CorefIOBHeuristicTagger:
    """a simple heuristic tagger for usage as a baseline"""

    def __init__(self, config):
        lang = config.get("lang", "en-us").split("-")[0]
        self.lang = lang
        res = f"{dirname(dirname(__file__))}/res/{self.lang}/corefiob.json"
        if not isfile(res):
            raise ValueError(f"unsupported language: {self.lang}")
        with open(res, "r") as f:
            data = json.load(f)
        self.joiner_tokens = data["joiner"]
        self.prev_toks = data["prev"]
        self.male_toks = data["male"]
        self.female_toks = data["female"]
        self.inanimate_toks = data["inanimate"]
        self.human_tokens = data["human"]
        self.neutral_coref_toks = data["neutral_coref"]
        self.male_coref_toks = data["male_coref"]
        self.female_coref_toks = data["female_coref"]
        self.inanimate_coref_toks = data["inanimate_coref"]

    def _tag_entities(self, iob):
        ents = {}

        valid_helper_tags = ["ADJ", "DET", "NUM"]
        valid_noun_tags = ["NOUN", "PROPN"]
        valid_tags = valid_noun_tags + valid_helper_tags + ["ADP"]

        for idx, (token, ptag, tag) in enumerate(iob):
            # the last token can never be a valid coreference entity
            if idx == len(iob) - 1:
                break
            is_plural = token.endswith("s")
            clean_token = token.lower().rstrip("s ")

            prev = iob[idx - 1] if idx > 0 else ("", "", "")
            prev2 = iob[idx - 2] if idx > 1 else ("", "", "")
            nxt = iob[idx + 1] if idx + 1 < len(iob) else ("", "", "")
            nxt2 = iob[idx + 2] if idx + 2 < len(iob) else ("", "", "")

            is_noun = ptag in valid_noun_tags and prev[0] not in self.joiner_tokens
            # plurals of the format NOUN and NOUN
            is_conjunction = token in self.joiner_tokens and \
                             prev[1] in valid_noun_tags and \
                             nxt[1] in valid_noun_tags
            # nouns of the form "NOUN of NOUN" or "NOUN of the ADJ NOUN"
            is_adp = ptag == "ADP" and "ENTITY" in prev[2] and nxt[1] in valid_noun_tags

            if is_adp:
                newtag = prev[2].replace("B-", "I-")
                iob[idx] = (token, ptag, newtag)
                iob[idx + 1] = (nxt[0], nxt[1], newtag)
                ents[idx] = ents[idx + 1] = newtag

            elif is_conjunction:
                iob[idx - 1] = (prev[0], prev[1], CorefIOBTags.ENTITY_PLURAL)
                iob[idx] = (token, ptag, CorefIOBTags.ENTITY_PLURAL_I)
                iob[idx + 1] = (nxt[0], nxt[1], CorefIOBTags.ENTITY_PLURAL_I)

                ents[idx - 1] = CorefIOBTags.ENTITY_PLURAL
                ents[idx] = CorefIOBTags.ENTITY_PLURAL_I
                ents[idx + 1] = CorefIOBTags.ENTITY_PLURAL_I
            elif is_noun:
                first = True
                # join multi word nouns
                if prev[1] == ptag:
                    t = prev[2].replace("B-", "I-")
                    iob[idx] = (token, ptag, t)
                    ents[idx] = t
                    continue

                # include adjectives and determinants
                if prev[1] in valid_helper_tags or \
                        prev[0].lower() in self.prev_toks:
                    first = False

                # implicitly gendered words, eg sister/brother mother/father
                if clean_token in self.female_toks:
                    if first:
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_FEMALE)
                        ents[idx] = CorefIOBTags.ENTITY_FEMALE
                    else:
                        iob[idx - 1] = (prev[0], prev[1], CorefIOBTags.ENTITY_FEMALE)
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_FEMALE_I)
                        ents[idx - 1] = CorefIOBTags.ENTITY_FEMALE
                        ents[idx] = CorefIOBTags.ENTITY_FEMALE_I
                elif clean_token in self.male_toks:
                    if first:
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_MALE)
                        ents[idx] = CorefIOBTags.ENTITY_MALE
                    else:
                        iob[idx - 1] = (prev[0], prev[1], CorefIOBTags.ENTITY_MALE)
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_MALE_I)
                        ents[idx - 1] = CorefIOBTags.ENTITY_MALE
                        ents[idx] = CorefIOBTags.ENTITY_MALE_I

                # known reference inanimate token, eg, iot device types "light"
                elif clean_token in self.inanimate_toks:
                    if first:
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_INANIMATE)
                        ents[idx] = CorefIOBTags.ENTITY_INANIMATE
                    elif prev2[1] in valid_helper_tags:
                        iob[idx - 2] = (prev2[0], prev2[1], CorefIOBTags.ENTITY_INANIMATE)
                        iob[idx - 1] = (prev[0], prev[1], CorefIOBTags.ENTITY_INANIMATE_I)
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_INANIMATE_I)

                        ents[idx - 2] = CorefIOBTags.ENTITY_INANIMATE
                        ents[idx - 1] = CorefIOBTags.ENTITY_INANIMATE_I
                        ents[idx] = CorefIOBTags.ENTITY_INANIMATE_I
                    else:
                        iob[idx - 1] = (prev[0], prev[1], CorefIOBTags.ENTITY_INANIMATE)
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_INANIMATE_I)

                        ents[idx - 1] = CorefIOBTags.ENTITY_INANIMATE
                        ents[idx] = CorefIOBTags.ENTITY_INANIMATE_I

                # ends with "s" its a plural noun
                elif is_plural:
                    if first:
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_PLURAL)
                        ents[idx] = CorefIOBTags.ENTITY_PLURAL
                    else:
                        iob[idx - 1] = (prev[0], prev[1], CorefIOBTags.ENTITY_PLURAL)
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_PLURAL_I)
                        ents[idx - 1] = CorefIOBTags.ENTITY_PLURAL
                        ents[idx] = CorefIOBTags.ENTITY_PLURAL_I

                # if its a unknown noun, its a neutral entity
                else:
                    if first:
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_NEUTRAL)
                        ents[idx] = CorefIOBTags.ENTITY_NEUTRAL
                    elif prev2[1] in valid_helper_tags:
                        iob[idx - 2] = (prev2[0], prev2[1], CorefIOBTags.ENTITY_NEUTRAL)
                        iob[idx - 1] = (prev[0], prev[1], CorefIOBTags.ENTITY_NEUTRAL_I)
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_NEUTRAL_I)

                        ents[idx - 2] = CorefIOBTags.ENTITY_NEUTRAL
                        ents[idx - 1] = CorefIOBTags.ENTITY_NEUTRAL_I
                        ents[idx] = CorefIOBTags.ENTITY_NEUTRAL_I
                    else:
                        iob[idx - 1] = (prev[0], prev[1], CorefIOBTags.ENTITY_NEUTRAL)
                        iob[idx] = (token, ptag, CorefIOBTags.ENTITY_NEUTRAL_I)

                        ents[idx - 1] = CorefIOBTags.ENTITY_NEUTRAL
                        ents[idx] = CorefIOBTags.ENTITY_NEUTRAL_I

                # handle sequential NOUN words
                if nxt[1] in valid_tags and nxt2[1] in valid_tags:
                    t = tag.replace("B-", "I-")
                    iob[idx + 1] = (nxt[0], nxt[1], t)
                    iob[idx + 2] = (nxt2[0], nxt2[1], t)
                    ents[idx + 1] = ents[idx + 2] = t

            elif ptag == "ADP" and prev[2] != "O":
                # handle NOUN of the NOUN
                if nxt[1] in valid_tags and nxt2[1] in valid_noun_tags:
                    t = prev[2].replace("B-", "I-")
                    iob[idx] = (token, ptag, t)
                    iob[idx + 1] = (nxt[0], nxt[1], t)
                    iob[idx + 2] = (nxt2[0], nxt2[1], t)
                    ents[idx] = ents[idx + 1] = ents[idx + 2] = t

        return iob, ents

    def _tag_prons(self, iob, ents):
        prons = {}
        for idx, (token, tag, _) in enumerate(iob):
            clean_token = token.lower().strip()
            if clean_token in self.inanimate_coref_toks:
                iob[idx] = (token, tag, CorefIOBTags.COREF_INANIMATE)
                prons[idx] = CorefIOBTags.COREF_INANIMATE
            elif clean_token in self.female_coref_toks:
                iob[idx] = (token, tag, CorefIOBTags.COREF_FEMALE)
                prons[idx] = CorefIOBTags.COREF_FEMALE
            elif clean_token in self.male_coref_toks:
                iob[idx] = (token, tag, CorefIOBTags.COREF_MALE)
                prons[idx] = CorefIOBTags.COREF_MALE
            elif clean_token in self.neutral_coref_toks:
                has_plural = any(v == CorefIOBTags.ENTITY_PLURAL for e, v in ents.items())
                if has_plural:
                    iob[idx] = (token, tag, CorefIOBTags.COREF_PLURAL)
                    prons[idx] = CorefIOBTags.COREF_PLURAL
                else:
                    iob[idx] = (token, tag, CorefIOBTags.COREF_NEUTRAL)
                    prons[idx] = CorefIOBTags.COREF_NEUTRAL
        return iob, prons

    def _untag_bad_candidates(self, iob, ents, bad_ents):
        # untag impossible entity corefs
        for e in bad_ents:
            if e in ents:
                ents.pop(e)
            token, ptag, _ = iob[e]
            iob[e] = (token, ptag, "O")
        return iob, ents

    def _disambiguate(self, iob, ents, prons):

        valid_helper_tags = ["ADJ", "DET", "NUM"]
        valid_noun_tags = ["NOUN", "PROPN"]

        # untag entities that can not possibly corefer
        # if there is no pronoun after the entity, then nothing can corefer to it
        bad_ents = [idx for idx in ents.keys() if not any(i > idx for i in prons.keys())]

        for ent, tag in ents.items():
            if ent in bad_ents:
                continue
            possible_coref = {k: v for k, v in prons.items() if k > ent}
            token, ptag, _ = iob[ent]
            prevtoken, prevptag, prevtag = iob[ent - 1]
            prev2 = iob[ent - 2] if ent > 1 else ("", "", "")
            clean_token = token.lower().rstrip("s ")

            neutral_corefs = any(t.endswith("NEUTRAL") for t in possible_coref.values())
            inanimate_corefs = any(t.endswith("INANIMATE") for t in possible_coref.values())
            female_corefs = {k: t for k, t in possible_coref.items() if t.endswith("-FEMALE")}
            male_corefs = {k: t for k, t in possible_coref.items() if t.endswith("-MALE")}

            # disambiguate neutral
            if tag.endswith("ENTITY-NEUTRAL") and ptag in valid_noun_tags:
                is_human = clean_token in self.human_tokens or ptag in ["PROPN"]

                # disambiguate neutral/inanimate
                if not neutral_corefs and inanimate_corefs and not is_human:
                    if tag.startswith("I-") or prevtag in [tag, CorefIOBTags.ENTITY_INANIMATE,
                                                           CorefIOBTags.ENTITY_INANIMATE_I]:
                        tag = CorefIOBTags.ENTITY_INANIMATE_I
                        if prev2[1] in valid_helper_tags:
                            iob[ent - 2] = (prev2[0], prev2[1], CorefIOBTags.ENTITY_INANIMATE)
                            iob[ent - 1] = (prevtoken, prevptag, CorefIOBTags.ENTITY_INANIMATE_I)
                            ents[ent - 2] = CorefIOBTags.ENTITY_INANIMATE
                            ents[ent - 1] = CorefIOBTags.ENTITY_INANIMATE_I
                        else:
                            iob[ent - 1] = (prevtoken, prevptag, CorefIOBTags.ENTITY_INANIMATE)
                            ents[ent - 1] = CorefIOBTags.ENTITY_INANIMATE
                    else:
                        tag = CorefIOBTags.ENTITY_INANIMATE
                    iob[ent] = (token, ptag, tag)
                    ents[ent] = tag

                elif is_human:
                    if male_corefs and not female_corefs:
                        if tag.startswith("I-") or prevtag in [tag, CorefIOBTags.ENTITY_MALE,
                                                               CorefIOBTags.ENTITY_MALE_I]:
                            tag = CorefIOBTags.ENTITY_MALE_I
                            if prevtag not in [CorefIOBTags.ENTITY_MALE, CorefIOBTags.ENTITY_MALE_I]:
                                iob[ent - 1] = (prevtoken, prevptag, CorefIOBTags.ENTITY_MALE)
                                ents[ent - 1] = CorefIOBTags.ENTITY_MALE
                        else:
                            tag = CorefIOBTags.ENTITY_MALE
                        iob[ent] = (token, ptag, tag)
                        ents[ent] = tag
                    elif female_corefs and not male_corefs:
                        if tag.startswith("I-") or prevtag in [tag, CorefIOBTags.ENTITY_MALE,
                                                               CorefIOBTags.ENTITY_MALE_I]:
                            tag = CorefIOBTags.ENTITY_FEMALE_I
                            iob[ent - 1] = (prevtoken, prevptag, CorefIOBTags.ENTITY_FEMALE)
                            ents[ent - 1] = CorefIOBTags.ENTITY_FEMALE
                        else:
                            tag = CorefIOBTags.ENTITY_FEMALE
                        iob[ent] = (token, ptag, tag)
                        ents[ent] = tag

                if (prevptag in valid_noun_tags or prevptag in valid_helper_tags or prevptag == "ADP") and \
                        (prev2[1] in valid_helper_tags or prev2[1] in valid_noun_tags):
                    iob[ent - 1] = (prevtoken, prevptag, tag.replace("B-", "I-"))
                    ents[ent - 1] = tag.replace("B-", "I-")
                    iob[ent] = (token, ptag, tag.replace("B-", "I-"))
                    ents[ent] = tag.replace("B-", "I-")

        iob, ents = self._untag_bad_candidates(iob, ents, bad_ents)

        return iob, ents, prons

    def _fix_iob_seqs(self, iob):
        valid_helper_tags = ["ADJ", "DET", "NUM", "ADP"]
        for idx, (token, ptag, tag) in enumerate(iob):
            if tag in ["O", CorefIOBTags.COREF_MALE, CorefIOBTags.COREF_FEMALE,
                       CorefIOBTags.COREF_INANIMATE, CorefIOBTags.COREF_NEUTRAL,
                       CorefIOBTags.COREF_PLURAL, CorefIOBTags.COREF_PLURAL_FEMALE, CorefIOBTags.COREF_PLURAL_MALE]:
                continue

            prev = iob[idx - 1] if idx > 0 else ("", "", "O")
            nxt = iob[idx + 1] if idx + 1 < len(iob) else ("", "", "O")

            # fix sequential B-ENTITY B-ENTITY -> B-ENTITY I-ENTITY
            if tag.startswith("B-"):
                if prev[2][2:] == tag[2:]:
                    iob[idx] = (token, ptag, tag.replace("B-", "I-"))

            # fix trailing not-nouns
            if ptag in valid_helper_tags:
                if nxt[2] == "O":
                    iob[idx] = (token, ptag, "O")
        return iob

    def _filter_coref_mismatches(self, iob, ents, prons):
        # untag mismatched entities with coref gender
        bad_ents = []
        for ent, tag in ents.items():
            possible_coref = {k: v for k, v in prons.items() if k > ent}
            token, ptag, _ = iob[ent]
            prevtoken, prevptag, _ = iob[ent - 1]

            neutral_corefs = any(t.endswith("NEUTRAL") for t in possible_coref.values())
            inanimate_corefs = any(t.endswith("INANIMATE") for t in possible_coref.values())
            plural_corefs = any(t.endswith("PLURAL") for t in possible_coref.values())

            female_corefs = {k: t for k, t in possible_coref.items() if t.endswith("-FEMALE")}
            male_corefs = {k: t for k, t in possible_coref.items() if t.endswith("-MALE")}

            # untag plural entities if there are no plural corefs
            if tag.endswith("ENTITY-PLURAL") and not plural_corefs:
                bad_ents.append(ent)
            # untag male entities if there are no male corefs
            elif tag.endswith("ENTITY-MALE") and not male_corefs:
                bad_ents.append(ent)
            # untag female entities if there are no female corefs
            elif tag.endswith("ENTITY-FEMALE") and not female_corefs:
                bad_ents.append(ent)
            # untag neutral entities
            # if there are no neutral corefs AND there are inanimate corefs
            elif tag.endswith("ENTITY-NEUTRAL") and \
                    not neutral_corefs and \
                    (inanimate_corefs or male_corefs or
                     female_corefs or plural_corefs):
                bad_ents.append(ent)

        iob, ents = self._untag_bad_candidates(iob, ents, bad_ents)
        return iob, ents

    def tag(self, postagged_toks):

        # failures to ignore
        # ("ohn called himJ", "John called him")  # John called John
        # ("John sent him his tax forms", "John sent him John tax forms")  # John sent John John tax forms

        # difficulty level: HARD
        # "John yelled at Jeff because he said he went back on his promise to fix his machines before he went home"
        # "John yelled at Jeff because Jeff said John went back on John promise to fix Jeff machines before John went home"
        # "John yelled at Jeff because Jeff said John went back on John promise to fix Jeff machines before Jeff went home"
        # "John yelled at Jeff because Jeff said John went back on John promise to fix Jeff machines before John went home"
        # "John yelled at Jeff because Jeff said John went back on John promise to fix John machines before Jeff went home"
        # "John yelled at Jeff because Jeff said John went back on John promise to fix John machines before John went home"
        # ("John yelled at Jeff because he said he went back on his promise to fix his machines before he went home",
        # "John yelled at Jeff because Jeff said John went back on John promise to fix Jeff machines before John went home")
        # Jeff Jeff Jeff Jeff Jeff Jeff ...

        iob = [(token, tag, "O") for (token, tag) in postagged_toks]

        iob, ents = self._tag_entities(iob)
        iob, prons = self._tag_prons(iob, ents)
        iob, ents, prons = self._disambiguate(iob, ents, prons)
        iob, ents = self._filter_coref_mismatches(iob, ents, prons)
        iob = self._fix_iob_seqs(iob)
        return iob

    @staticmethod
    def normalize_corefs(iobtagged_tokens):
        sentences = []
        for toks in iobtagged_tokens:
            ents = {}
            s = ""
            for t, _, iob in toks:
                if iob == "O":
                    s += t + " "
                elif "B-ENTITY" in iob:
                    s += t + " "
                    ents[iob.replace("B-", "")] = t
                elif "I-ENTITY" in iob:
                    s += t + " "
                    ents[iob.replace("I-", "")] = t
                elif "B-COREF" in iob:
                    i = iob.replace("B-COREF-", "ENTITY-")
                    if i in ents:
                        s += ents[i] + " "
                    else:
                        s += t + " "

            sentences.append(s.strip())
        return sentences
