
| pipeline | language | accuracy | params | size (MB)|
|----------|----------|----------|--------|----------|
| cv2_lemma_1704447165 | en | 0.9681827240764956| {'solver': 'lbfgs', 'learning_rate': 'adaptive', 'hidden_layer_sizes': (233,), 'early_stopping': True, 'alpha': 0.03, 'activation': 'logistic'} | 1.465 |
| cv2_lemma_1704447238 | en | 0.9746417362522987| {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': (30, 100), 'early_stopping': False, 'alpha': 0.0385, 'activation': 'tanh'} | 1.066 |
| cv2_lemma_1704447308 | en | 0.971692711114536| {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': (20, 48), 'early_stopping': False, 'alpha': 0.05, 'activation': 'relu'} | 0.559 |
| cv2_lemma_1704447400 | en | 0.9740416762462981| {'solver': 'adam', 'learning_rate': 'invscaling', 'hidden_layer_sizes': (102, 149, 38), 'early_stopping': False, 'alpha': 0.005, 'activation': 'identity'} | 4.938 |
| cv2_lemma_1704447489 | en | 0.9738275013386414| {'solver': 'lbfgs', 'learning_rate': 'invscaling', 'hidden_layer_sizes': (31, 123), 'early_stopping': False, 'alpha': 0.0105, 'activation': 'tanh'} | 0.408 |