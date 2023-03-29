class HeuristicUtteranceTagger:

    def __init__(self, config=None):
        self.config = config or {}
        lang = self.config.get("lang", "en-us")
        self.lang = lang.split("-")[0]
        if lang not in ["en", "es", "pt", "uk", "de", "nl", "fr", "it", "sv", "no"]:
            raise ValueError(f"unsupported language: {lang}")

    def predict(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        return [self.classify(s, self.lang) for s in sentences]

    # TODO - refactor to use resource files under /res
    @staticmethod
    def _get_kwords(lang):
        # TODO - review, non-english generated via ChatGPT
        if lang == "en":
            command_action_keywords = ["do", "make", "perform", "carry out", "execute", "accomplish", "achieve", "boil",
                                       "bring", "brush", "call", "turn on", "turn off"]
            command_denial_keywords = ["don't", "do not", "prohibited", "strictly prohibited", "stay away", "stop",
                                       "not allowed"]
            question_query_keywords = ["how", "what", "where", "when", "which", "who", "why", "tell me about"]
            question_request_keywords = ["would you mind", "could you", "can you", "please"]
            question_yesno_keywords = ["is", "are", "do", "does", "will", "can", "could", "may", "might", "should",
                                       "would"]
            sentence_exclamation_keywords = ["what", "how", "amazing", "impressive"]
            sentence_social_keywords = ["congratulations", "enjoy", "get well soon", "good luck", "good morning"]
        elif lang == "pt":
            command_action_keywords = ["fazer", "executar", "realizar", "cozinhar", "ferver", "trazer", "chamar",
                                       "ligar", "desligar", "faça", "execute", "vá", "traga"]
            command_denial_keywords = ["não faça", "proibido", "estritamente proibido", "afaste-se", "pare",
                                       "não permitido", "evite"]
            question_query_keywords = ["como", "o que", "onde", "quando", "qual", "quem", "por que", "me fale sobre"]
            question_request_keywords = ["você se importaria", "você poderia", "você pode", "por favor",
                                         "poderia", "por favor", "me dê", "me traga"]
            question_yesno_keywords = ["é", "são", "faz", "fazem", "vai", "pode", "poderia",
                                       "deveria", "gostaria", "está", "tem", "há", "você é", "você está"]
            sentence_exclamation_keywords = ["o que", "como", "incrível", "fantástico", "impressionante", "maravilhoso"]
            sentence_social_keywords = ["parabéns", "aproveite", "melhoras", "boa sorte", "bom dia",
                                        "bom trabalho", "feliz aniversário"]
        elif lang == "es":
            command_action_keywords = ["hacer", "realizar", "ejecutar", "cocinar", "hervir", "traer", "llamar",
                                       "encender", "apagar"]
            command_denial_keywords = ["no", "prohibido", "estrictamente prohibido", "mantenerse alejado", "parar",
                                       "no permitido"]
            question_query_keywords = ["cómo", "qué", "dónde", "cuándo", "cuál", "quién", "por qué", "háblame de"]
            question_request_keywords = ["te importaría", "podrías", "puedes", "por favor",
                                         "dame", "tráeme"]
            question_yesno_keywords = ["es", "son", "hace", "hacen", "irá", "puede", "podría",
                                       "debería", "gustaría", "está", "tiene", "hay", "eres", "estás"]
            sentence_exclamation_keywords = ["qué", "cómo", "increíble", "impresionante", "maravilloso"]
            sentence_social_keywords = ["felicidades", "disfruta", "mejórate pronto", "buena suerte", "buenos días",
                                        "buen trabajo", "feliz cumpleaños"]
        elif lang == "fr":
            command_action_keywords = ["faire", "réaliser", "accomplir", "cuisiner", "bouillir", "amener", "appeler",
                                       "allumer", "éteindre"]
            command_denial_keywords = ["ne fais pas", "prohibé", "strictement interdit", "éloigne-toi", "arrête",
                                       "pas autorisé", "évite"]
            question_query_keywords = ["comment", "quoi", "où", "quand", "quel", "qui", "pourquoi", "parle-moi de"]
            question_request_keywords = ["pourrais-tu", "peux-tu", "s'il te plaît", "pourrais-je", "pourrais-je avoir",
                                         "apporte-moi", "donne-moi"]
            question_yesno_keywords = ["est-ce que", "suis-je", "suis-je en train", "sont-ils", "est-ce que tu es",
                                       "es-tu", "seras-tu", "peux-tu", "pourrais-tu", "devrais-je", "voudrais-tu",
                                       "as-tu", "as-tu besoin", "y a-t-il", "y a-t-il des", "c'est", "ce sont",
                                       "ça va", "ça ne va pas", "est-ce grave"]
            sentence_exclamation_keywords = ["quoi", "comment", "incroyable", "fantastique", "impressionnant",
                                             "merveilleux"]
            sentence_social_keywords = ["félicitations", "profites-en", "rétablis-toi vite", "bonne chance", "bonjour",
                                        "bon travail", "joyeux anniversaire"]
        elif lang == "uk":
            command_action_keywords = ["зробити", "виконати", "виконувати", "готувати", "кип'ятити", "принести",
                                       "викликати",
                                       "вмикати", "вимикати", "робити", "виконуйте", "йти", "принесіть"]
            command_denial_keywords = ["не робити", "заборонено", "строго заборонено", "уникати", "зупинитися",
                                       "не дозволяється", "забороняється"]
            question_query_keywords = ["як", "що", "де", "коли", "який", "хто", "чому", "розкажіть мені про"]
            question_request_keywords = ["ви не проти", "ви могли б", "ви можете", "будь ласка",
                                         "можливо", "будь ласка", "дайте мені", "принесіть мені"]
            question_yesno_keywords = ["чи", "вони", "робить", "роблять", "йде", "може", "міг би",
                                       "повинен", "хотів би", "є", "має", "є", "ви", "ти"]
            sentence_exclamation_keywords = ["як", "що", "неймовірно", "фантастично", "вражаюче", "чудово"]
            sentence_social_keywords = ["вітаємо", "насолоджуйтесь", "одужуйте швидко", "вдалих починань",
                                        "доброго ранку", "гарно працюєте", "з днем народження"]
        elif lang == "nl":
            command_action_keywords = ["doe", "maak", "uitvoeren", "uitvoer", "uitvoering", "bereiden", "koken",
                                       "brengen", "bel", "aanzetten", "uitzetten"]
            command_denial_keywords = ["niet doen", "verboden", "absoluut verboden", "wegblijven", "stop",
                                       "niet toegestaan"]
            question_query_keywords = ["hoe", "wat", "waar", "wanneer", "welke", "wie", "waarom", "vertel me over"]
            question_request_keywords = ["zou je", "kun je", "kan je", "graag"]
            question_yesno_keywords = ["is", "zijn", "doen", "maakt", "zal", "kan", "kunnen", "mogen", "zou", "moeten"]
            sentence_exclamation_keywords = ["wat", "hoe", "geweldig", "indrukwekkend"]
            sentence_social_keywords = ["gefeliciteerd", "genieten", "beterschap", "veel succes", "goedemorgen",
                                        "goed werk", "fijne verjaardag"]
        elif lang == "de":
            command_action_keywords = ["machen", "ausführen", "erledigen", "kochen", "bringen", "anrufen",
                                       "einschalten", "ausschalten", "tu", "mach", "geh", "bringe"]
            command_denial_keywords = ["nicht machen", "verboten", "streng verboten", "fernbleiben", "stopp",
                                       "nicht erlaubt", "vermeiden"]
            question_query_keywords = ["wie", "was", "wo", "wann", "welche", "wer", "warum", "erzähl mir von"]
            question_request_keywords = ["würdest du", "könntest du", "kannst du", "bitte", "bring mir", "gib mir"]
            question_yesno_keywords = ["ist", "sind", "macht", "machen", "wird", "kann", "könnte", "sollte",
                                       "würde", "bist", "du bist"]
            sentence_exclamation_keywords = ["was", "wie", "unglaublich", "fantastisch", "beeindruckend"]
            sentence_social_keywords = ["herzlichen glückwunsch", "genieße es", "gute besserung", "viel glück",
                                        "guten morgen", "gute arbeit", "alles gute zum geburtstag"]
        elif lang == "it":
            command_action_keywords = ["fai", "esegui", "realizza", "cuoci", "porta", "chiamare", "accendere",
                                       "spegnere",
                                       "fa", "esegui", "vai", "porta"]
            command_denial_keywords = ["non fare", "proibito", "strettamente proibito", "stai lontano", "fermati",
                                       "non permesso", "evita"]
            question_query_keywords = ["come", "cosa", "dove", "quando", "quale", "chi", "perché", "dimmi di"]
            question_request_keywords = ["ti dispiacerebbe", "potresti", "puoi", "per favore", "potrebbe",
                                         "per piacere", "dammi", "portami"]
            question_yesno_keywords = ["è", "sono", "fai", "fanno", "andrai", "puoi", "potresti", "dovresti",
                                       "vorresti", "sei", "hai", "c'è", "tu sei", "tu stai"]
            sentence_exclamation_keywords = ["che", "come", "incredibile", "fantastico", "impressionante",
                                             "meraviglioso"]
            sentence_social_keywords = ["complimenti", "goditi", "mi raccomando", "in bocca al lupo", "buongiorno",
                                        "buon lavoro", "buon compleanno"]
        elif lang == "sv":
            command_action_keywords = ["gör", "utför", "genomför", "koka", "ta med", "borsta", "ring", "starta",
                                       "stoppa"]
            command_denial_keywords = ["gör inte", "förbjudet", "strikst förbjudet", "håll dig borta", "stoppa",
                                       "inte tillåtet"]
            question_query_keywords = ["hur", "vad", "var", "när", "vilken", "vem", "varför", "berätta för mig om"]
            question_request_keywords = ["skulle du vilja", "kan du", "kan du snälla", "snälla"]
            question_yesno_keywords = ["är", "är det", "är de", "gör", "gör det", "gör de", "vill", "kan", "kunde",
                                       "borde", "skulle"]
            sentence_exclamation_keywords = ["vad", "hur", "fantastiskt", "imponerande"]
            sentence_social_keywords = ["grattis", "njut av", "krya på dig", "lycka till", "god morgon"]
        elif lang == "no":
            command_action_keywords = ["gjør", "lage", "utføre", "koke", "ta med", "ring", "skru på", "skru av"]
            command_denial_keywords = ["ikke gjør", "forbudt", "strengt forbudt", "hold deg unna", "stopp",
                                       "ikke tillatt"]
            question_query_keywords = ["hvordan", "hva", "hvor", "når", "hvilken", "hvem", "hvorfor", "fortell meg om"]
            question_request_keywords = ["ville du kunne", "kan du", "kunne du", "vær så snill", "gi meg", "ta med"]
            question_yesno_keywords = ["er", "er det", "er det noen", "er det mulig", "gjør", "vil", "kan", "kunne",
                                       "skal",
                                       "bør", "må", "skulle", "ville"]
            sentence_exclamation_keywords = ["hva", "så flott", "fantastisk", "imponerende"]
            sentence_social_keywords = ["gratulerer", "kos deg", "god bedring", "lykke til", "god morgen"]
        else:
            raise ValueError(f"unsupported language: {lang}")

        return command_action_keywords, command_denial_keywords, question_query_keywords, \
               question_request_keywords, question_yesno_keywords, sentence_exclamation_keywords, \
               sentence_social_keywords

    @classmethod
    def classify(cls, sentence, lang):
        sentence = sentence.lower().strip()
        command_action_keywords, command_denial_keywords, question_query_keywords, \
        question_request_keywords, question_yesno_keywords, sentence_exclamation_keywords, \
        sentence_social_keywords = cls._get_kwords(lang)

        if any(sentence.startswith(w) for w in command_action_keywords):
            return "COMMAND:ACTION"
        elif any(sentence.startswith(w) for w in question_yesno_keywords):
            return "QUESTION:YESNO"
        elif any(sentence.startswith(w) for w in question_query_keywords):
            return "QUESTION:QUERY"
        elif any(w in sentence for w in command_denial_keywords):
            return "COMMAND:DENIAL"
        elif any(w in sentence for w in question_request_keywords):
            return "QUESTION:REQUEST"
        elif any(w in sentence for w in sentence_social_keywords):
            return "SENTENCE:SOCIAL"
        elif any(w in sentence for w in sentence_exclamation_keywords):
            return "SENTENCE:EXCLAMATION"
        else:
            return "SENTENCE:STATEMENT"


if __name__ == "__main__":
    sentence = "How much does it cost?"
    label = HeuristicUtteranceTagger.classify(sentence, "en")
    print(label)  # Output: "QUESTION:QUERY"

    sentence = "tem cerveja no frigorifico?"
    label = HeuristicUtteranceTagger.classify(sentence, "pt")
    print(label)  # Output: "QUESTION:YESNO"

    sentence = "quem inventou o telefone"
    label = HeuristicUtteranceTagger({"lang": "pt"}).predict(sentence)
    print(label)  # Output: "QUESTION:QUERY"
