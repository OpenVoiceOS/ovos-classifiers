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