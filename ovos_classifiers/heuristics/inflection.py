# -*- coding: utf-8 -*-
"""
English implementation taken from
~~~~~~~~~~~~
inflection __version__ = '0.5.1'

A port of Ruby on Rails' inflector to Python.

:copyright: (c) 2012-2020 by Janne Vanhala

:license: MIT, see LICENSE for more details.
"""
import enum
import re
import typing
import unicodedata


class PluralCategory(str, enum.Enum):
    """
    plural category for the specified amount. Category can be one of
    the categories specified by Unicode CLDR Plural Rules.
    For more details:
    http://cldr.unicode.org/index/cldr-spec/plural-rules
    https://unicode-org.github.io/cldr-staging/charts/37/supplemental/language_plural_rules.html
    """
    CARDINAL = "cardinal"
    ORDINAL = "ordinal"
    RANGE = "range"


class PluralAmount(str, enum.Enum):
    """
    For more details:
    http://cldr.unicode.org/index/cldr-spec/plural-rules
    https://unicode-org.github.io/cldr-staging/charts/37/supplemental/language_plural_rules.html
    """
    ZERO = "zero"
    ONE = "one"
    TWO = "two"
    FEW = "few"
    MANY = "many"
    OTHER = "other"


class GrammaticalGender(str, enum.Enum):
    """
    Gender in European languages:
    - Light blue: no gender system.
    - common/neuter.
    - masculine/feminine.
    - animate/inanimate.
    - masculine/feminine/neuter.

    For more details:
    https://en.wikipedia.org/wiki/Grammatical_gender
    """
    NONE = "none"  # langs without grammatical gender
    COMMON = "common"
    NEUTER = "neuter"
    MASCULINE = "masculine"
    FEMININE = "feminine"
    ANIMATE = "animate"
    INANIMATE = "inanimate"


class Inflection:
    def __init__(self, lang):
        self.lang = lang

    def ordinal(self, number: int, gender=GrammaticalGender.NONE) -> str:
        """
        Return the suffix that should be added to a number to denote the position
        in an ordered sequence such as 1st, 2nd, 3rd, 4th.
        """
        if self.lang.startswith("en"):
            return self.ordinal_en(number, gender)
        elif self.lang.startswith("pt"):
            return self.ordinal_pt(number, gender)
        elif self.lang.startswith("de"):
            return f"{number}."
        raise NotImplementedError

    def get_plural_category(self, amount, ptype=PluralCategory.CARDINAL):
        """
        Get plural category for the specified amount. Category can be one of
        the categories specified by Unicode CLDR Plural Rules.
        For more details:
        http://cldr.unicode.org/index/cldr-spec/plural-rules
        https://unicode-org.github.io/cldr-staging/charts/37/supplemental/language_plural_rules.html
        Args:
            amount(int or float or pair or list): The amount that is used to
                determine the category. If type is range, it must contain
                the start and end numbers.
            ptype(str): Either cardinal (default), ordinal or range.
        Returns:
            (str): The plural category. Either zero, one, two, few, many or other.
        """
        if self.lang.startswith("en"):
            return self.get_plural_category_en(amount, ptype)
        elif self.lang.startswith("sl"):
            return self.get_plural_category_sl(amount, ptype)
        elif self.lang.startswith("de"):
            return self.get_plural_category_de(amount, ptype)
        elif ptype == PluralCategory.CARDINAL:
            if amount == 1:
                return PluralAmount.ONE
            else:
                return PluralAmount.OTHER
        raise NotImplementedError

    def get_plural_form(self, word, amount, ptype=PluralCategory.CARDINAL):
        """
        Get plural form of the specified word for the specified amount.
        Args:
            word(str): Word to be pluralized.
            amount(int or float or pair or list): The amount that is used to
                determine the category. If type is range, it must contain
                the start and end numbers.
            ptype(str): Either cardinal (default), ordinal or range.
        Returns:
            (str): Pluralized word.
        """
        if self.lang.startswith("en"):
            return self.get_plural_form_en(word, amount, ptype)
        if self.lang.startswith("pt"):
            return self.get_plural_form_pt(word, amount, ptype)
        elif self.lang.startswith("de"):
            return self.get_plural_form_de(word, amount, ptype)
        raise NotImplementedError

    # generic lang agnostic helpers
    def ordinalize(self, number: int) -> str:
        """
        Turn a number into an ordinal string used to denote the position in an
        ordered sequence such as 1st, 2nd, 3rd, 4th.

        Examples::

            >>> ordinalize(1)
            '1st'
            >>> ordinalize(2)
            '2nd'
            >>> ordinalize(1002)
            '1002nd'
            >>> ordinalize(1003)
            '1003rd'
            >>> ordinalize(-11)
            '-11th'
            >>> ordinalize(-1021)
            '-1021st'

        """
        return "{}{}".format(number, self.ordinal(number))

    @staticmethod
    def parameterize(string: str, separator: str = '-') -> str:
        """
        Replace special characters in a string so that it may be used as part of a
        'pretty' URL.

        Example::

            >>> Inflection.parameterize(u"Donald E. Knuth")
            'donald-e-knuth'

        """
        string = Inflection.transliterate(string)
        # Turn unwanted chars into the separator
        string = re.sub(r"(?i)[^a-z0-9\-_]+", separator, string)
        if separator:
            re_sep = re.escape(separator)
            # No more than one of the separator in a row.
            string = re.sub(r'%s{2,}' % re_sep, separator, string)
            # Remove leading/trailing separator.
            string = re.sub(r"(?i)^{sep}|{sep}$".format(sep=re_sep), '', string)

        return string.lower()

    def tableize(self, word: str) -> str:
        """
        Create the name of a table like Rails does for models to table names. This
        method uses the :func:`pluralize` method on the last word in the string.

        Examples::

            >>> tableize('RawScaledScorer')
            'raw_scaled_scorers'
            >>> tableize('egg_and_ham')
            'egg_and_hams'
            >>> tableize('fancyCategory')
            'fancy_categories'
        """
        return self.pluralize(self.underscore(word))

    @staticmethod
    def camelize(string: str, uppercase_first_letter: bool = True) -> str:
        """
        Convert strings to CamelCase.

        Examples::

            >>> Inflection.camelize("device_type")
            'DeviceType'
            >>> Inflection.camelize("device_type", False)
            'deviceType'

        :func:`camelize` can be thought of as a inverse of :func:`underscore`,
        although there are some cases where that does not hold::

            >>> Inflection.camelize(Inflection.underscore("IOError"))
            'IoError'

        :param uppercase_first_letter: if set to `True` :func:`camelize` converts
            strings to UpperCamelCase. If set to `False` :func:`camelize` produces
            lowerCamelCase. Defaults to `True`.
        """
        if uppercase_first_letter:
            return re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), string)
        else:
            return string[0].lower() + Inflection.camelize(string)[1:]

    @staticmethod
    def dasherize(word: str) -> str:
        """Replace underscores with dashes in the string.

        Example::

            >>> Inflection.dasherize("puni_puni")
            'puni-puni'

        """
        return word.replace('_', '-')

    @staticmethod
    def humanize(word: str) -> str:
        """
        Capitalize the first word and turn underscores into spaces and strip a
        trailing ``"_id"``, if any. Like :func:`titleize`, this is meant for
        creating pretty output.

        Examples::

            >>> Inflection.humanize("employee_salary")
            'Employee salary'
            >>> Inflection.humanize("author_id")
            'Author'

        """
        word = re.sub(r"_id$", "", word)
        word = word.replace('_', ' ')
        word = re.sub(r"(?i)([a-z\d]*)", lambda m: m.group(1).lower(), word)
        word = re.sub(r"^\w", lambda m: m.group(0).upper(), word)
        return word

    @staticmethod
    def titleize(word: str) -> str:
        """
        Capitalize all the words and replace some characters in the string to
        create a nicer looking title. :func:`titleize` is meant for creating pretty
        output.

        Examples::

          >>> Inflection.titleize("man from the boondocks")
          'Man From The Boondocks'
          >>> Inflection.titleize("x-men: the last stand")
          'X Men: The Last Stand'
          >>> Inflection.titleize("TheManWithoutAPast")
          'The Man Without A Past'
          >>> Inflection.titleize("raiders_of_the_lost_ark")
          'Raiders Of The Lost Ark'

        """
        return re.sub(
            r"\b('?\w)",
            lambda match: match.group(1).capitalize(),
            Inflection.humanize(Inflection.underscore(word)).title()
        )

    @staticmethod
    def transliterate(string: str) -> str:
        """
        Replace non-ASCII characters with an ASCII approximation. If no
        approximation exists, the non-ASCII character is ignored. The string must
        be ``unicode``.

        Examples::

            >>> Inflection.transliterate('älämölö')
            'alamolo'
            >>> Inflection.transliterate('Ærøskøbing')
            'rskbing'

        """
        normalized = unicodedata.normalize('NFKD', string)
        return normalized.encode('ascii', 'ignore').decode('ascii')

    @staticmethod
    def underscore(word: str) -> str:
        """
        Make an underscored, lowercase form from the expression in the string.

        Example::

            >>> Inflection.underscore("DeviceType")
            'device_type'

        As a rule of thumb you can think of :func:`underscore` as the inverse of
        :func:`camelize`, though there are cases where that does not hold::

            >>> Inflection.camelize(Inflection.underscore("IOError"))
            'IoError'

        """
        word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
        word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
        word = word.replace("-", "_")
        return word.lower()

    # portuguese specific handlers
    @staticmethod
    def _get_pt_data():
        _VOWELS_PT = ["a", "ã", "á", "à",
                      "e", "é", "è",
                      "i", "ì", "í",
                      "o", "ó", "ò", "õ",
                      "u", "ú", "ù"]

        _INVARIANTS_PT = ["ontem", "depressa", "ali", "além", "sob", "por", "contra", "desde", "entre",
                          "até", "perante", "porém", "contudo", "todavia", "entretanto", "senão", "portanto",
                          "oba", "eba", "exceto", "excepto", "apenas", "menos", "também", "inclusive", "aliás",
                          "que", "onde", "isto", "isso", "aquilo", "algo", "alguém", "nada", "ninguém", "tudo", "cada",
                          "outrem", "quem", "mais", "menos", "demais",
                          # NOTE some words ommited because it depends on POS_TAG
                          # NOTE these multi word expressions are also invariant
                          "ou melhor", "isto é", "por exemplo", "a saber", "digo", "ou seja",
                          "por assim dizer", "com efeito", "ou antes"]

        _PLURAL_EXCEPTIONS_PT = {
            "cânon": "cânones",
            "cós": "coses",  # cós (unchanged word) is also valid
            "cais": "cais",
            "xis": "xis",
            "mal": "males",
            "cônsul": "cônsules",
            "mel": "méis",  # "meles" also valid
            "fel": "féis",  # "feles" also valid
            "cal": "cais",  # "cales" also valid
            "aval": "avais",  # "avales also valid
            "mol": "móis",  # "moles also valid
            "real": "réis",
            "fax": "faxes",
            "cálix": "cálices",
            "índex": "índices",
            "apêndix": "apêndices",
            "hélix": "hélices",
            "hálux": "háluces",
            "códex": "códices",
            "fénix": "fénixes",  # "fénix" also valid
            "til": "tis",  # "tiles" also valid
            "pão": "pães",
            "cão": "cães",
            "alemão": "alemães",
            "balão": "balões",
            "anão": "anões",
            "dez": "dez",
            "três": "três",
            "seis": "seis"
        }

        # in general words that end with "s" in singular form should be added below
        _SINGULAR_EXCEPTIONS_PT = {v: k for k, v in _PLURAL_EXCEPTIONS_PT.items()}

        return _VOWELS_PT, _INVARIANTS_PT, _PLURAL_EXCEPTIONS_PT, _SINGULAR_EXCEPTIONS_PT

    def _singularize_pt(self, word):
        _VOWELS_PT, _INVARIANTS_PT, _PLURAL_EXCEPTIONS_PT, _SINGULAR_EXCEPTIONS_PT = self._get_pt_data()
        if word in _INVARIANTS_PT:
            return word
        if word in _SINGULAR_EXCEPTIONS_PT:
            return _SINGULAR_EXCEPTIONS_PT[word]
        # TODO implement is_plural helper
        # can not ensure word is in plural, assuming it is,
        # if in singular form it might in some cases be wrongly mutated
        # in general words that end with "s" in singular form should be added to exceptions dict
        if word.endswith("is"):
            return word.rstrip("is") + "il"
        if word.endswith("ões"):
            return word.replace("ões", "ão")
        if word.endswith("ães"):
            return word.replace("ães", "ão")
        if word.endswith("es"):
            return word.rstrip("es")
        if word.endswith("s"):
            return word.rstrip("s")
        return word

    def _pluralize_pt(self, word):
        _VOWELS_PT, _INVARIANTS_PT, _PLURAL_EXCEPTIONS_PT, _SINGULAR_EXCEPTIONS_PT = self._get_pt_data()
        if word in _INVARIANTS_PT:
            return word
        if word in _PLURAL_EXCEPTIONS_PT:
            return _PLURAL_EXCEPTIONS_PT[word]
        if word.endswith("x"):
            return word
        if word.endswith("s"):
            # TODO - this will catch too many words, need a better check
            # if word[-2] in _VOWELS_PT or word[-3] in _VOWELS_PT:
            # if word is an oxytone, add "es", else word remains unchanged
            # https://en.wikipedia.org/wiki/Oxytone
            #    return word + "es"
            return word
        if word.endswith("ão"):
            # crap, can either end with "ãos", "aẽs" or "ões", most times they are all valid
            # the other times lets hope the word is in exceptions dict
            # TODO check if numeric, then it's always "ões"
            return word + "s"
        if word[-1] in _VOWELS_PT:
            # if word ends with a vowel add an "s"
            return word + 's'
        for ending in ["r", "z", "n"]:
            if word.endswith(ending):
                return word + "es"
        for ending in ["al", "el", "ol", "ul"]:
            if word.endswith(ending):
                return word.rstrip("l") + "is"
        if word.endswith("il"):
            return word.rstrip("l") + "s"
        if word.endswith("m"):
            return word.rstrip("m") + "ns"
        # foreign words that have been "unportuguesified" have an "s" added
        # simple check is looking for endings that don't exist in portuguese
        for ending in ["w", "y", "k", "t"]:
            if word.endswith(ending):
                return word + "s"
        return word

    @staticmethod
    def ordinal_pt(number: int, gender: GrammaticalGender.MASCULINE) -> str:
        """
        Return the suffix that should be added to a number to denote the position
        in an ordered sequence such as 1st, 2nd, 3rd, 4th.
        """
        if gender == GrammaticalGender.FEMININE:
            return "ª"
        else:
            return "º"

    def get_plural_form_pt(self, word, amount, ptype=PluralCategory.CARDINAL):
        """
            Get plural form of the specified word for the specified amount.
            Args:
                word(str): Word to be pluralized.
                amount(int or float or pair or list): The amount that is used to
                    determine the category. If type is range, it must contain
                    the start and end numbers.
                ptype(str): Either cardinal (default), ordinal or range.
            Returns:
                (str): Pluralized word.
            """
        if amount == 1:
            return self._singularize_pt(word)
        return self._pluralize_pt(word)

    # german specific handlers
    @staticmethod
    def _get_de_data():
        # _VOWELS_DE = ["a", "e", "i", "o", "u", "y"]
        # _INVARIANTS_DE = []
        _PLURALS = [
            ("aal", "äle"), ("aat", "aaten"), ("abe", "aben" ), (r"(.*(?:b|kr))ach$", r'\1äche'),
            ("lach", "lache"), ("ach", "ächer"), ("ade", "aden"), ("aden", "äden"),
            ("age", "agen"), (r"(.*(?:h|k|z))ahn$", r'\1ähne'), ("ahn", "ahnen"),
            ("fahr", "fahren" ), ("ahr", "ahre" ), ("fakt", "fakten"), ("akt", "akte"),
            ("akte", "akten"), ("ale", "alen"), ("ame", "amen"), ("amt", "ämter"),
            ("ane", "anen"), ("ang", "änge"), ("tank", "tanks"), ("ank", "änke"),
            ("ann", "änner"), ("ant", "anten"), ("aph", "aphen"), ("are", "aren"),
            ("arn", "arne"), ("ase", "asen"), ("ate", "aten"), ("statt", "stätten"),
            ("att", "ätter"), ("atz", "atzen"), ("aum", "äume"), (r"(.*(?:m|l))aus$", r'\1äuse'),
            ("aus", "äuser"), ("bad", "bäder"), ("bel", "beln"), ("bot", "bote"),
            ("che", "chen"), ("chs", "chse"), ("cke", "cken"), (r"(.*(?:na|n|u))del$", r'\1deln'),
            ("ader", "adern"), ("nder", "nde"), (r"(.*(?:w|r))ebe$", r'\1eben'),
            ("ede", "eden"), ("ehl", "ehle"), ("ehr", "ehre"), ("eil", "eile"),
            ("eim", "eime"), ("eis", "eise"), (r"(.*(?:tr|ch))eit$", r'\1eite'),
            ("eit", "eiten"), (r"(.*(?:s|t))ekt$", r'\1ekten'), ("ekt", "ekte"),
            ("held", "helden"), ("eld", "elder"), ("ell", "elle" ), ("ene", "enen"),
            ("enz", "enzen"), ("erd", "erde"), ("ere", "eren"), ("erk", "erke"),
            ("ern", "erne"), ("ert", "erte"), ("ese", "esen"), (r"(.*(?:n|t|d))ess$", r'\1essen'),
            ("ess", "esse"), ("nest", "nester"), ("test", "tests"), ("est", "este"),
            ("etz", "etze"), ("eug", "euge"), ("eur", "eure"), (r"(.*(?:ta|au))fel$", r'\1feln'),
            ("ffel", "ffeln"), (r"(.*(?:if|ie|ng))fer$", r'\1fern'), ("ffe", "ffen"),
            (r"(.*(?:re|in|ku))gel$", r'\1geln'), ("iger", "ige"), ("gie", "gien"),
            (r"(.*(?:lic|disc))her$", r'\1he'), ("hie", "hien" ), ("hle", "hlen"),
            ("hme", "hmen"), ("hne", "hnen"), ("hof", "höfe"), ("hre", "hren"),
            ("hrt", "hrten"), ("hse", "hsen"), ("hte", "hten"), ("ich", "iche"),
            (r"(.*(?:tr|kl|st))ick$", r'\1icks'), ("ick", "icke"), ("ide", "iden"),
            ("ieb", "iebe"), ("ief", "iefe"), ("ieg", "iege"), ("iel", "iele"),
            (r"(.*(?:l|r))ien$", r'\1ien'), ("ien", "ium"), ("iet", "iete"),
            ("ife", "ifen"), ("iff", "iffe"), (r"(.*(?:g|st|l))ift$", r'\1ifte'),
            ("ift", "iften"), ("ige", "igen"), ("ika", "ikas"), ("ild", "ilder"),
            ("ilm", "ilme"), ("ine", "inen"), (r"(.*(?:l|r))ing$", r'\1inge'),
            ("ing", "ings"), ("pion", "pione"), ("ion", "ionen"), ("ise", "isen"),
            ("iss", "isse"), ("geist", "geister"), ("ist", "isten"), ("ite", "iten"),
            ("itt", "itte"), ("itz", "itze"), ("ium", "ien"), ("lag", "läge"),
            ("lan", "läne"), ("lar", "lare"), ("lei", "leien"), ("llen", "llen"),
            ("len", "lene"), (r"(.*(?:u|i))eller$", r'\1elle'), ("lge", "lgen"),
            ("lie", "lien"), ("lle", "llen"), ("mmel", "mmel"), ("mel", "meln"),
            (r"(.*(?:a|u))mmer$", r'\1mmern'), ("mme", "mmen"), ("mpe", "mpen"),
            ("mpf", "mpfe"), ("mus", "men"), ("gnat", "gnaten"), ("nat", "nate"),
            ("ände", "ände"), ("nde", "nden"), (r"(.*(?:r|s))ener$", r'\1ene'),
            (r"(^(?![gG]e).*)nge$", r'\1ngen'), ("nge", "ngen"), ("nie", "nien"),
            ("nis", "nisse"), ("nke", "nken"), ("nkt", "nkte"), ("nne", "nnen"),
            ("nst", "nste"), ("nte", "nten"), ("nze", "nzen"), ("ock", "öcke"),
            ("ode", "oden"), ("off", "offe"), ("oge", "ogen"), ("ohn", "öhne"),
            ("rohr", "rohre"), ("ohr", "ohren"), ("olz", "ölzer"), ("one", "onen"),
            ("oot", "oote"), ("opf", "öpfe"), ("ord", "orde"), ("orm", "ormen"),
            ("orn", "örner"), ("ose", "osen"), ("ote", "oten"), (r"(.*(?:am|p))pel$", r'\1peln'),
            ("pie", "pien"), ("ppe", "ppen"), ("rag", "räge"), ("frau", "frauen"),
            ("rau", "raün"), (r"(^(?![gG]e).*)rbe$", r'\1rben'), ("rde", "rden"),
            (r"(.*(?:sch|b))rei$", r'\1reie'), ("rei", "reien"), ("rie", "rien"),
            ("rin", "rinnen"), ("rke", "rken"), ("rot", "rote"), ("rre", "rren"),
            ("rte", "rten"), ("ruf", "rufe"), ("rzt", "rzte"), ("mensch", "menschen"),
            ("sch", "sche"), (r"(.*(?:lo|fa))ser$", r'\1sern'), ("sie", "sien"),
            ("sik", "siken"), ("sse", "ssen"), ("ste", "sten"), ("tag", "tage"),
            (r"(.*(?:is|an|ch))tel$", r'\1teln'), (r"(.*(?:tig|tes|ag|am|hr))ter$", r'\1te'),
            ("tie", "tien"), ("tin", "tinnen"), ("tiv", "tive"), ("tor", "toren"),
            ("tte", "tten"), ("datum", "daten"), ("tum", "ta"), ("tum", "tümer"),
            ("tur", "turen"), ("tze", "tzen"), ("ube", "uben"), ("äude", "äude"),
            ("ude", "uden"), ("ufe", "ufen"), ("uge", "ugen"), ("uhr", "uhren"),
            ("ule", "ulen"), ("ume", "umen"), (r"(.*(?:pr|w))ung$", r'\1ünge'),
            ("ung", "ungen"), ("äuse", "äuse"), ("use", "usen"), ("uss", "üsse"),
            ("eute", "eute"), ("ute", "uten"), ("weg", "wege"), ("zug", "züge"),
            ("ück", "ücke")
        ]

        _SINGULAR = [tuple(reversed(tup)) for tup in _PLURALS]
        for i, tup in enumerate(_SINGULAR):
            if r"(" in tup[1]:
                rule = re.search(r"(.*(?:\(.*\)))", tup[1])[0]
                _SINGULAR[i] = (tup[0].replace(r"\1", rule)+"$",
                                tup[1].replace(rule, r"\1").replace("$", ""))

        return _PLURALS, _SINGULAR

    def _singularize_de(self, word):
        _, SINGULAR = Inflection._get_de_data()
        for end, replacement in SINGULAR:
            if not "(" in end:
                if word.endswith(end):
                    return word.replace(end, replacement)
            elif re.search(end, word):
                return re.sub(end, replacement, word)
        
        return word

    def _pluralize_de(self, word):
        PLURALS, _ = Inflection._get_de_data()
        for end, replacement in PLURALS:
            if not "(" in end:
                if word.endswith(end):
                    return word.replace(end, replacement)
            elif re.search(end, word):
                return re.sub(end, replacement, word)
        
        return word

    def get_plural_form_de(self, word, amount, ptype=PluralCategory.CARDINAL):
        """
            Get plural form of the specified word for the specified amount.
            Args:
                word(str): Word to be pluralized.
                amount(int or float or pair or list): The amount that is used to
                    determine the category. If type is range, it must contain
                    the start and end numbers.
                ptype(str): Either cardinal (default), ordinal or range.
            Returns:
                (str): Pluralized word.
            """
        if amount == 1:
            return self._singularize_de(word)
        return self._pluralize_de(word)
    
    def get_plural_category_de(self, amount, ptype=PluralCategory.CARDINAL):
        if type == PluralCategory.CARDINAL:
            if amount == 1:
                return PluralAmount.ONE
            else:
                return PluralAmount.OTHER

        elif type == PluralCategory.ORDINAL:
            return PluralAmount.OTHER

        elif type == PluralCategory.RANGE:
            if not (isinstance(amount, tuple) or isinstance(amount, list)) or len(amount) != 2:
                raise ValueError("Argument \"number\" must be tuple|list type with the start and end numbers")

            return PluralAmount.OTHER

        else:
            return ValueError("Argument \"type\" must be cardinal|ordinal|range")

    # english specific handlers
    @staticmethod
    def _get_en_data():
        RegexReplaceList = typing.List[typing.Tuple[str, str]]

        PLURALS: RegexReplaceList = [
            (r"(?i)(quiz)$", r'\1zes'),
            (r"(?i)^(oxen)$", r'\1'),
            (r"(?i)^(ox)$", r'\1en'),
            (r"(?i)(m|l)ice$", r'\1ice'),
            (r"(?i)(m|l)ouse$", r'\1ice'),
            (r"(?i)(passer)s?by$", r'\1sby'),
            (r"(?i)(matr|vert|ind)(?:ix|ex)$", r'\1ices'),
            (r"(?i)(x|ch|ss|sh)$", r'\1es'),
            (r"(?i)([^aeiouy]|qu)y$", r'\1ies'),
            (r"(?i)(hive)$", r'\1s'),
            (r"(?i)([lr])f$", r'\1ves'),
            (r"(?i)([^f])fe$", r'\1ves'),
            (r"(?i)sis$", 'ses'),
            (r"(?i)([ti])a$", r'\1a'),
            (r"(?i)([ti])um$", r'\1a'),
            (r"(?i)(buffal|potat|tomat)o$", r'\1oes'),
            (r"(?i)(bu)s$", r'\1ses'),
            (r"(?i)(alias|status)$", r'\1es'),
            (r"(?i)(octop|vir)i$", r'\1i'),
            (r"(?i)(octop|vir)us$", r'\1i'),
            (r"(?i)^(ax|test)is$", r'\1es'),
            (r"(?i)s$", 's'),
            (r"$", 's'),
        ]

        SINGULARS: RegexReplaceList = [
            (r"(?i)(database)s$", r'\1'),
            (r"(?i)(quiz)zes$", r'\1'),
            (r"(?i)(matr)ices$", r'\1ix'),
            (r"(?i)(vert|ind)ices$", r'\1ex'),
            (r"(?i)(passer)sby$", r'\1by'),
            (r"(?i)^(ox)en", r'\1'),
            (r"(?i)(alias|status)(es)?$", r'\1'),
            (r"(?i)(octop|vir)(us|i)$", r'\1us'),
            (r"(?i)^(a)x[ie]s$", r'\1xis'),
            (r"(?i)(cris|test)(is|es)$", r'\1is'),
            (r"(?i)(shoe)s$", r'\1'),
            (r"(?i)(o)es$", r'\1'),
            (r"(?i)(bus)(es)?$", r'\1'),
            (r"(?i)(m|l)ice$", r'\1ouse'),
            (r"(?i)(x|ch|ss|sh)es$", r'\1'),
            (r"(?i)(m)ovies$", r'\1ovie'),
            (r"(?i)(s)eries$", r'\1eries'),
            (r"(?i)([^aeiouy]|qu)ies$", r'\1y'),
            (r"(?i)([lr])ves$", r'\1f'),
            (r"(?i)(tive)s$", r'\1'),
            (r"(?i)(hive)s$", r'\1'),
            (r"(?i)([^f])ves$", r'\1fe'),
            (r"(?i)(t)he(sis|ses)$", r"\1hesis"),
            (r"(?i)(s)ynop(sis|ses)$", r"\1ynopsis"),
            (r"(?i)(p)rogno(sis|ses)$", r"\1rognosis"),
            (r"(?i)(p)arenthe(sis|ses)$", r"\1arenthesis"),
            (r"(?i)(d)iagno(sis|ses)$", r"\1iagnosis"),
            (r"(?i)(b)a(sis|ses)$", r"\1asis"),
            (r"(?i)(a)naly(sis|ses)$", r"\1nalysis"),
            (r"(?i)([ti])a$", r'\1um'),
            (r"(?i)(n)ews$", r'\1ews'),
            (r"(?i)(ss)$", r'\1'),
            (r"(?i)s$", ''),
        ]

        UNCOUNTABLES: typing.Set[str] = {
            'equipment',
            'fish',
            'information',
            'jeans',
            'money',
            'rice',
            'series',
            'sheep',
            'species'}

        def _irregular(singular: str, plural: str) -> None:
            """
            A convenience function to add appropriate rules to plurals and singular
            for irregular words.

            :param singular: irregular word in singular form
            :param plural: irregular word in plural form
            """

            def caseinsensitive(string: str) -> str:
                return ''.join('[' + char + char.upper() + ']' for char in string)

            if singular[0].upper() == plural[0].upper():
                PLURALS.insert(0, (
                    r"(?i)({}){}$".format(singular[0], singular[1:]),
                    r'\1' + plural[1:]
                ))
                PLURALS.insert(0, (
                    r"(?i)({}){}$".format(plural[0], plural[1:]),
                    r'\1' + plural[1:]
                ))
                SINGULARS.insert(0, (
                    r"(?i)({}){}$".format(plural[0], plural[1:]),
                    r'\1' + singular[1:]
                ))
            else:
                PLURALS.insert(0, (
                    r"{}{}$".format(singular[0].upper(),
                                    caseinsensitive(singular[1:])),
                    plural[0].upper() + plural[1:]
                ))
                PLURALS.insert(0, (
                    r"{}{}$".format(singular[0].lower(),
                                    caseinsensitive(singular[1:])),
                    plural[0].lower() + plural[1:]
                ))
                PLURALS.insert(0, (
                    r"{}{}$".format(plural[0].upper(), caseinsensitive(plural[1:])),
                    plural[0].upper() + plural[1:]
                ))
                PLURALS.insert(0, (
                    r"{}{}$".format(plural[0].lower(), caseinsensitive(plural[1:])),
                    plural[0].lower() + plural[1:]
                ))
                SINGULARS.insert(0, (
                    r"{}{}$".format(plural[0].upper(), caseinsensitive(plural[1:])),
                    singular[0].upper() + singular[1:]
                ))
                SINGULARS.insert(0, (
                    r"{}{}$".format(plural[0].lower(), caseinsensitive(plural[1:])),
                    singular[0].lower() + singular[1:]
                ))

        _irregular('person', 'people')
        _irregular('man', 'men')
        _irregular('human', 'humans')
        _irregular('child', 'children')
        _irregular('sex', 'sexes')
        _irregular('move', 'moves')
        _irregular('cow', 'kine')
        _irregular('zombie', 'zombies')

        return PLURALS, SINGULARS, UNCOUNTABLES

    def _pluralize_en(self, word: str) -> str:
        """
        Return the plural form of a word.

        Examples::

            >>> pluralize("posts")
            'posts'
            >>> pluralize("octopus")
            'octopi'
            >>> pluralize("sheep")
            'sheep'
            >>> pluralize("CamelOctopus")
            'CamelOctopi'

        """
        PLURALS, SINGULARS, UNCOUNTABLES = self._get_en_data()
        if not word or word.lower() in UNCOUNTABLES:
            return word
        else:
            for rule, replacement in PLURALS:
                if re.search(rule, word):
                    return re.sub(rule, replacement, word)
            return word

    def _singularize_en(self, word: str) -> str:
        """
        Return the singular form of a word, the reverse of :func:`pluralize`.

        Examples::

            >>> singularize("posts")
            'post'
            >>> singularize("octopi")
            'octopus'
            >>> singularize("sheep")
            'sheep'
            >>> singularize("word")
            'word'
            >>> singularize("CamelOctopi")
            'CamelOctopus'

        """
        PLURALS, SINGULARS, UNCOUNTABLES = self._get_en_data()
        for inflection in UNCOUNTABLES:
            if re.search(r'(?i)\b(%s)\Z' % inflection, word):
                return word

        for rule, replacement in SINGULARS:
            if re.search(rule, word):
                return re.sub(rule, replacement, word)
        return word

    @staticmethod
    def ordinal_en(number: int, gender: GrammaticalGender.NONE) -> str:
        """
        Return the suffix that should be added to a number to denote the position
        in an ordered sequence such as 1st, 2nd, 3rd, 4th.

        Examples::

            >>> ordinal(1)
            'st'
            >>> ordinal(2)
            'nd'
            >>> ordinal(1002)
            'nd'
            >>> ordinal(1003)
            'rd'
            >>> ordinal(-11)
            'th'
            >>> ordinal(-1021)
            'st'

        """
        number = abs(int(number))
        if number % 100 in (11, 12, 13):
            return "th"
        else:
            return {
                1: "st",
                2: "nd",
                3: "rd",
            }.get(number % 10, "th")

    def get_plural_category_en(self, amount, ptype=PluralCategory.CARDINAL):
        if type == PluralCategory.CARDINAL:
            if amount == 1:
                return PluralAmount.ONE
            else:
                return PluralAmount.OTHER

        elif type == PluralCategory.ORDINAL:
            if amount % 10 == 1 and amount % 100 != 11:
                return PluralAmount.ONE
            elif amount % 10 == 2 and amount % 100 != 12:
                return PluralAmount.TWO
            elif amount % 10 == 3 and amount % 100 != 13:
                return PluralAmount.FEW
            else:
                return PluralAmount.OTHER

        elif type == PluralCategory.RANGE:
            if not (isinstance(amount, tuple) or isinstance(amount, list)) or len(amount) != 2:
                raise ValueError("Argument \"number\" must be tuple|list type with the start and end numbers")

            return PluralAmount.OTHER

        else:
            return ValueError("Argument \"type\" must be cardinal|ordinal|range")

    def get_plural_form_en(self, word, amount, ptype=PluralCategory.CARDINAL):
        """
        Get plural form of the specified word for the specified amount.
        Args:
            word(str): Word to be pluralized.
            amount(int or float or pair or list): The amount that is used to
                determine the category. If type is range, it must contain
                the start and end numbers.
            ptype(str): Either cardinal (default), ordinal or range.
        Returns:
            (str): Pluralized word.
        """
        if amount == 1:
            return self._singularize_en(word)
        else:
            return self._pluralize_en(word)

    # sl specific handlers
    def get_plural_category_sl(self, amount, type=PluralCategory.CARDINAL):
        if type == PluralCategory.CARDINAL:
            if amount % 100 == 1 and amount % 1 == 0:
                return PluralAmount.ONE
            elif amount % 100 == 2 and amount % 1 == 0:
                return PluralAmount.TWO
            elif amount % 100 == 3 or amount % 100 == 4 or amount % 1 != 0:
                return PluralAmount.FEW
            else:
                return PluralAmount.OTHER

        elif type == PluralCategory.ORDINAL:
            return PluralAmount.OTHER

        elif type == PluralCategory.RANGE:
            if not (isinstance(amount, tuple) or isinstance(amount, list)) or len(amount) != 2:
                raise ValueError("Argument \"number\" must be tuple|list type with the start and end numbers")

            end = self.get_plural_category_sl(amount[1])

            if end == PluralAmount.ONE or end == PluralAmount.FEW:
                return PluralAmount.FEW
            elif end == PluralAmount.TWO:
                return PluralAmount.TWO
            elif end == PluralAmount.OTHER:
                return PluralAmount.OTHER

        else:
            return ValueError("Argument \"type\" must be cardinal|ordinal|range")
