from quebra_frases import word_tokenize as _wtok
import re


def word_tokenize(utterance, lang=None):
    if lang is not None and lang.startswith("pt"):
        return word_tokenize_pt(utterance)
    elif lang is not None and lang.startswith("ca"):
        return word_tokenize_ca(utterance)
    # Split things like 12%
    utterance = re.sub(r"([0-9]+)([\%])", r"\1 \2", utterance)
    # Split thins like #1
    utterance = re.sub(r"(\#)([0-9]+\b)", r"\1 \2", utterance)
    return _wtok(utterance)


def word_tokenize_pt(utterance):
    # Split things like 12%
    utterance = re.sub(r"([0-9]+)([\%])", r"\1 \2", utterance)
    # Split things like #1
    utterance = re.sub(r"(\#)([0-9]+\b)", r"\1 \2", utterance)
    # Split things like amo-te
    utterance = re.sub(r"([a-zA-Z]+)(-)([a-zA-Z]+\b)", r"\1 \2 \3",
                       utterance)
    tokens = utterance.split()
    if tokens[-1] == '-':
        tokens = tokens[:-1]

    return tokens


def word_tokenize_ca(utterance):
    # Split things like 12%
    utterance = re.sub(r"([0-9]+)([\%])", r"\1 \2", utterance)
    # Split things like #1
    utterance = re.sub(r"(\#)([0-9]+\b)", r"\1 \2", utterance)
    # Don't split at -
    tokens = utterance.split()
    if tokens[-1] == '-':
        tokens = tokens[:-1]

    return tokens
