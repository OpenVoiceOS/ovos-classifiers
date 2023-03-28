
def nilc_to_udep(tag):
    """
       # NILC tagset
           Adjetivo ADJ
           Advérbio ADV
           Artigo ART
           Número Cardinal NC
           Número Ordinal ORD
           Outros Números  NO
           Substantivo Comum N
           Nome Próprio NP
           Conj. Coordenativa  CONJCOORD
           Conj. Subordinativa  CONJSUB
           Pronome Demonstrativo PD
           Pronome Indefinido  PIND
           Pronome Oblíquo Átono  PPOA
           Pronome Pessoal Caso Reto  PPR
           Pronome Possessivo PPS
           Pronome Relativo  PR
           Pronome Oblíquo Tônico  PPOT
           Pronome Interrogativo  PINT
           Pronome Apassivador  PAPASS
           Pronome de Realce PREAL
           Pronome Tratamento  PTRA
           Preposição  PREP
           Verbo Auxiliar  VAUX
           Verbo de Ligação  VLIG
           Verbo Intransitivo VINT
           Verbo Transitivo Direto  VTD
           Verbo Transitivo Indireto  VTI
           Verbo Bitransitivo  VBI
           Interjeição I
           Locução Adverbial LADV
           Locução Conjuncional LCONJ
           Locução Prepositiva LPREP
           Locução Pronominal LP
           Palavra Denotativa PDEN
           Locução Denotativa LDEN
           Palavras ou Símbolos  Residuais RES

       # universal dependencies tagset
        ADJ: adjective
        ADP: adposition
        ADV: adverb
        AUX: auxiliary
        CCONJ: coordinating conjunction
        DET: determiner
        INTJ: interjection
        NOUN: noun
        NUM: numeral
        PART: particle
        PRON: pronoun
        PROPN: proper noun
        PUNCT: punctuation
        SCONJ: subordinating conjunction
        SYM: symbol
        VERB: verb
        X: other
       """
    tagdict = {
        'ADJ': 'ADJ',
        'ADV': 'ADV',
        'ART': 'DET',
        "I": 'INTJ',
        "PREP": "ADP",  # PREPOSITION
        "N": "NOUN",
        "NP": "PROPN",
        "NC": "NUM",  # NUMERIC CARDINAL
        "ORD": "NUM",  # ORDINAL
        "NO": "NUM",  # NUMERIC OTHER
        "CONJCOORD": "CCONJ",
        "CONJSUB": "SCONJ",
        "PD": 'PRON',
        "PIND": 'PRON',  # UNDEFINED PRONOUN
        "PPOA": 'PRON',
        "PPR": 'PRON',
        "PPS": 'PRON',
        "PR": 'PRON',
        "PPOT": 'PRON',
        "PINT": 'PRON',
        "PAPASS": 'PRON',
        "PREAL": 'PRON',
        'PTRA': 'PRON',

        # TODO are entries below accurate ?
        'PDEN': 'PART',
        'LDEN': 'AUX',
        'LP': 'AUX',
        'LPREP': 'AUX',
        'LCONJ': 'AUX',
        'LADV': 'AUX',
        'RES': 'SYM'  # RESIDUAL
    }
    tag = tag.split("+")[0].strip()
    if tag in tagdict:
        return tagdict.get(tag)

    tag_lower = tag.lower()
    if tag_lower in ['!', '"', "'", '(', ')', ',', '-', '.', '...',
                     ':', ';', '?', '[', ']']:
        return 'PUNCT'
    if tag_lower.startswith("v"):
        return 'VERB'
    if tag_lower.startswith("p"):
        return 'PRON'
    return 'X'


def eagles_to_udep(tag):
    """EAGLES http://www.ilc.cnr.it/EAGLES96/annotate/annotate.html"""
    tagdict = {
        'X': 'X',
        'Y': 'X',
        'i': 'X',
        'w': 'NOUN'  # time
    }
    if tag in tagdict:
        return tagdict[tag]
    tag = tag.lower().strip()
    if tag.startswith("v"):
        return 'VERB'
    if tag.startswith("p"):
        return 'PRON'
    if tag.startswith("n"):
        return 'NOUN'
    if tag.startswith("d"):
        return 'DET'
    if tag.startswith("a"):
        return 'ADJ'
    if tag.startswith("z"):
        return 'NUM'
    if tag.startswith("s"):
        return 'ADP'
    if tag.startswith("r"):
        return 'ADV'
    if tag.startswith("c"):
        return 'CONJ'
    if tag.startswith("f"):
        return '.'
    return tagdict.get(tag) or 'PART'

