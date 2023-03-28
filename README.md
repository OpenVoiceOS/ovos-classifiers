# OVOS Classifiers

## Usage

see `scripts/training` for training scripts


postag

```python
from ovos_classifiers.postag import OVOSPostag

p = OVOSPostag("en")
print(p.model_id)
print(p.tagset) # Universal Dependencies
print(p.postag("The brown fox jumped over the lazy dog"))
# [('The', 'DET'), ('brown', 'ADJ'), ('fox', 'NOUN'), ('jumped', 'VERB'), ('over', 'ADP'), ('the', 'DET'), ('lazy', 'ADJ'), ('dog', 'VERB')]

p = OVOSPostag("pt")
print(p.model_id)
print(p.tagset) # Universal Dependencies
print(p.postag("Ontem fui passear com o meu cão"))
# [('Ontem', ('Ontem', 'ADV')), ('fui', ('fui', 'VERB')), ('passear', ('passear', 'VERB')), ('com', ('com', 'ADP')), ('o', ('o', 'DET')), ('meu', ('meu', 'PRON')), ('cão', ('cão', 'NOUN'))]

p = OVOSPostag("nltk-brown-brown-ngram-postag")
print(p.model_id)
print(p.tagset) # brown
print(p.postag("The brown fox jumped over the lazy dog"))
# [('The', ('The', 'AT')), ('brown', ('brown', 'JJ')), ('fox', ('fox', 'NN')), ('jumped', ('jumped', 'VBD')), ('over', ('over', 'IN')), ('the', ('the', 'AT')), ('lazy', ('lazy', 'JJ')), ('dog', ('dog', 'NN'))]

p = OVOSPostag("nltk-floresta-visl-brill-postag")
print(p.model_id)
print(p.tagset) # VISL (Portuguese)
print(p.postag("Ontem fui passear com o meu cão"))
# [('Ontem', ('Ontem', 'adv')), ('fui', ('fui', 'v-fin')), ('passear', ('passear', 'v-inf')), ('com', ('com', 'prp')), ('o', ('o', 'art')), ('meu', ('meu', 'pron-det')), ('cão', ('cão', 'n'))]
```

utterance tags

```python
from ovos_classifiers.utttags import OVOSUtteranceTagger


sentences = [
    "The brown fox jumped over the lazy dog",
    "Turn off the TV",
    "Turn on the lights",
    "thats amazing",
    "what time is it",
    "tell me about einstein"
]
p = OVOSUtteranceTagger("en")
print(p.model_id)
print(p.tagset)
print(p.tag(sentences))
# ['SENTENCE:STATEMENT' 'COMMAND:ACTION' 'COMMAND:ACTION'
#  'SENTENCE:EXCLAMATION' 'QUESTION:QUERY' 'COMMAND:ACTION']


```