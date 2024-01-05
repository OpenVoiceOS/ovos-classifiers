| pipeline               | language | accuracy           | params                                                                              | size (MB) |
|------------------------|----------|--------------------|-------------------------------------------------------------------------------------|-----------|
| tfidf_lemma_1704442668 | en       | 0.8422402468412058 | {'penalty': 'l1', 'l1_ratio': 0.5, 'early_stopping': False, 'alpha': 0.0001}        | 0.084     |
| tfidf_lemma_1704442719 | en       | 0.8512445554525825 | {'penalty': 'elasticnet', 'l1_ratio': 0.7, 'early_stopping': True, 'alpha': 0.005}  | 0.085     |
| tfidf_lemma_1704442769 | en       | 0.8341548180629628 | {'penalty': 'elasticnet', 'l1_ratio': 0.9, 'early_stopping': True, 'alpha': 0.0001} | 0.084     |
| tfidf_lemma_1704442819 | en       | 0.8472131620506758 | {'penalty': 'l1', 'l1_ratio': 0.15, 'early_stopping': False, 'alpha': 0.002}        | 0.083     |
| tfidf_lemma_1704442869 | en       | 0.8461698528114111 | {'penalty': None, 'l1_ratio': 0.15, 'early_stopping': False, 'alpha': 0.05}         | 0.084     |