# OVOS Classifiers

WIP - open during construction, **pre-alpha**

built on top of nltk and scikit-learn

- provide baseline feature extraction pipelines suited for each task
- provide user facing classes for each NLP task needed in the OVOS ecosystem
  - transparently load different models (model_path or external plugin)
- provide dataset loaders suited for each of those tasks per language
- provide baseline heuristic implementation for each task per language
  - replaces ovos-lingua-franca  
- provide baseline implementations and benchmarks using classical nltk and scikit-learn algorithms
  - minimum viable implementation to ensure lang support

## Usage

see `scripts/training` for training scripts


## Module Structure

- datasets - dataset downloaders and loaders
  - tagsets - postag tagset conversion
  - wordnet - helpers to interact with Wordnet
- heuristics - implement solutions to common NLP problems without any machine learning
  - corefiob - coreference resolution
  - inflection - pluralization / singularization
  - keyword_extraction - extract keywords from text
  - lang_detect - language classifier
  - machine_comprehension - question answering (query + passage)
  - normalize - text normalization
  - numeric - extract numbers from text
  - phonemizer - text 2 ARPA phonemes
  - postag - regex postag + nltk default postag wrapper
  - summarization - text summarization
  - time - date / time / duration extraction from text
  - tokenize - tokenization utilities
  - utttags - question classifier
- opm - plugins for models trained with this package
  - heuristics - plugins using implementations from heuristics module
  - nltk - plugins built with nltk
- skovos - scikit learn based classifiers
  - features - feature extraction utilities
    - en - english specific
    - pt - portuguese specific
  - classifier - classification models
  - nn - neural network (MultiLayerPerceptron) models
  - pipelines - ready to use pipelines for classifier training
  - tagger - utterance tokens tagging models
- tasks - base classes for solving NLP problems
  - classifier - classify text
  - tagger - tag utterance tokens
- utils - misc utilities
- corefiob - coreference resolution entrypoint (ovos-config aware)
- postag - postag entrypoint (ovos-config aware)
- utttags - question classification entrypoint (ovos-config aware)

## Plugins

#### plugins for models trained with this package

these plugins use the best available model per language out of the box, take into account global ovos-config

**implements**: normalization, coreference solver, postag

```python
from ovos_classifiers.opm import OVOSPostagPlugin, OVOSCoreferenceSolverPlugin, UtteranceNormalizerPlugin, CoreferenceNormalizerPlugin

 print(OVOSPostagPlugin().postag("I like pizza", "en"))
# [(0, 1, 'I', 'PRON'), (2, 6, 'like', 'VERB'), (7, 12, 'pizza', 'NOUN')]

coref = OVOSCoreferenceSolverPlugin()
print(coref.solve_corefs("Mom is awesome, she said she loves me!", "en"))

u, _ = UtteranceNormalizerPlugin().transform(["Mom is awesome, she said she loves me!"])
print(u)
# ['Mom is awesome, she said she loves me!', 'Mom is awesome , she said she loves me']
u, _ = CoreferenceNormalizerPlugin().transform(u)  #
print(u)
# ['Mom is awesome , Mom said Mom loves me', 'Mom is awesome, she said she loves me!', 'Mom is awesome , she said she loves me']
```

#### plugins implemented with nltk

**implements**: keyword extraction, question answering, postag

```python
from ovos_classifiers.opm.nltk import NltkPostagPlugin, RakeExtractorPlugin, WordnetSolverPlugin

print(NltkPostagPlugin().postag("I like pizza", "en"))
# [(0, 1, 'I', 'PRON'), (2, 6, 'like', 'VERB'), (7, 12, 'pizza', 'NOUN')]

k = RakeExtractorPlugin()
k.extract("who invented the telephone", "en")  # {'telephone': 0.5, 'invented': 0.5}
k.extract("what is the speed of light", "en")  # {'speed': 0.5, 'light': 0.5}

d = WordnetSolverPlugin()
sentence = d.spoken_answer("what is the definition of computer")
print(sentence)
# a machine for performing calculations automatically

d = WordnetSolverPlugin()
sentence = d.spoken_answer("qual é a definição de computador", lang="pt")
print(sentence)
# uma máquina para realizar cálculos automaticamente
```

#### heuristics based plugins

**implements**: keyword extraction, question answering, summarization

```python
from ovos_classifiers.opm.heuristics import HeuristicKeywordExtractorPlugin, BM25SolverPlugin, HeuristicSummarizerPlugin

k = HeuristicKeywordExtractorPlugin()
k.extract("who invented the telephone", "en")  # {'telephone': 1.0}
k.extract("what is the speed of light", "en")  # {'speed of light': 1.0}


doc = """
Introducing OpenVoiceOS - The Free and Open-Source Personal Assistant and Smart Speaker.

OpenVoiceOS is a new player in the smart speaker market, offering a powerful and flexible alternative to proprietary solutions like Amazon Echo and Google Home.

With OpenVoiceOS, you have complete control over your personal data and the ability to customize and extend the functionality of your smart speaker.

Built on open-source software, OpenVoiceOS is designed to provide users with a seamless and intuitive voice interface for controlling their smart home devices, playing music, setting reminders, and much more.

The platform leverages cutting-edge technology, including machine learning and natural language processing, to deliver a highly responsive and accurate experience.

In addition to its voice capabilities, OpenVoiceOS features a touch-screen GUI made using QT5 and the KF5 framework.

The GUI provides an intuitive, user-friendly interface that allows you to access the full range of OpenVoiceOS features and functionality.

Whether you prefer voice commands or a more traditional touch interface, OpenVoiceOS has you covered.

One of the key advantages of OpenVoiceOS is its open-source nature, which means that anyone with the technical skills can contribute to the platform and help shape its future.

Whether you're a software developer, data scientist, or just someone with a passion for technology, you can get involved and help build the next generation of personal assistants and smart speakers.

With OpenVoiceOS, you have the option to run the platform fully offline, giving you complete control over your data and ensuring that your information is never shared with third parties. This makes OpenVoiceOS the perfect choice for anyone who values privacy and security.

So if you're looking for a personal assistant and smart speaker that gives you the freedom and control you deserve, be sure to check out OpenVoiceOS today!
"""

b = BM25SolverPlugin()
print(b.get_best_passage(doc, "does OpenVoiceOS run offline"))
# With OpenVoiceOS , you have the option to run the platform fully offline , giving you complete control over your data and ensuring that your information is never shared with third parties .

h = HeuristicSummarizerPlugin()
print(h.tldr(doc, lang="en"))
#     Built on open-source software, OpenVoiceOS is designed to provide users with a seamless and intuitive voice interface for controlling their smart home devices, playing music, setting reminders, and much more.
#     Whether you're a software developer, data scientist, or just someone with a passion for technology, you can get involved and help build the next generation of personal assistants and smart speakers.
#     With OpenVoiceOS, you have complete control over your personal data and the ability to customize and extend the functionality of your smart speaker.
#     With OpenVoiceOS, you have the option to run the platform fully offline, giving you complete control over your data and ensuring that your information is never shared with third parties.
#     So if you're looking for a personal assistant and smart speaker that gives you the freedom and control you deserve, be sure to check out OpenVoiceOS today!
#     One of the key advantages of OpenVoiceOS is its open-source nature, which means that anyone with the technical skills can contribute to the platform and help shape its future.
#     OpenVoiceOS is a new player in the smart speaker market, offering a powerful and flexible alternative to proprietary solutions like Amazon Echo and Google Home.

```
