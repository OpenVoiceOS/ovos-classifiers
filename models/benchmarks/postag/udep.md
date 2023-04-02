

## Benchmark - nltk Universal Dependencies corpus

datasets: brown+treebank+nps_chat+mte

dataset len: 78558


| Model | Training Accuracy | Accuracy 	|
|-------|----------|----------|
| nltk-brown-udep-brill-postag | 0.9621202214376008  | 0.9378531707666165 |
| nltk-brown-udep-ngram-postag | 0.9483806285689377  | 0.9265874881253237 |
| nltk-brown+treebank-udep-brill-postag | 0.9260809638545099  | 0.9063426558020686 |
| sklearn-brown+treebank-udep-lsvc-postag | 0.9231  | 0.9002 |
| nltk-brown+treebank-udep-ngram-postag | 0.914857689371472  | 0.8957456033879135 |
| sklearn-brown+treebank-udep-percep-postag | 0.914  | 0.8906 |
| sklearn-brown+treebank+mte+nps_chat-udep-nb-postag | 0.8463  | 0.8522 |
| sklearn-brown+treebank-udep-nb-postag | 0.8597  | 0.8408 |
| sklearn-brown+treebank-udep-dtree-postag | 0.8581  | 0.8279 |
| nltk-treebank-udep-brill-postag | 0.923858085525752  | 0.794122929736996 |
| nltk-treebank-udep-ngram-postag | 0.9213300197103437  | 0.7920994989132127 |
| nltk-nps_chat-udep-brill-postag | 0.8870742385195765  | 0.7652237911684682 |
| nltk-nps_chat-udep-ngram-postag | 0.8860185673921819  | 0.7651178487085544 |
| nltk-mte_en-udep-brill-postag | 0.9534966003138172  | 0.7349558198893988 |
| nltk-mte_en-udep-ngram-postag | 0.946020367350706  | 0.7331961928068578 |