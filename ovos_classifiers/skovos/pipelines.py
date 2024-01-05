from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from ovos_classifiers.skovos.features import WordFeaturesVectorizer, POSTaggerVectorizer, \
    PronounTaggerVectorizer, CorefIOBTaggerVectorizer, SingleWordFeaturesVectorizer, TokenizerTransformer, \
    SkipGramVectorizer, LangFeaturesVectorizer, OCPKeywordFeaturesVectorizer, \
    ClassifierProbaVectorizer
from ovos_classifiers.skovos.features.en import QuestionFeaturesVectorizerEN, WordNetLemmatizerTransformer

"""
DEFAULT_PIPELINES = {
    "en": FeatureUnion([
        ("cv2", CountVectorizer(ngram_range=(1, 2))),
        ('tfidf_lemma', Pipeline([
            ("lemma", LemmatizerTransformerEN()),
            ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
        ])),
        ("postag", POSTaggerVectorizerEN())
    ]),
    "pt": FeatureUnion([
        ("cv2", CountVectorizer(ngram_range=(1, 2))),
        #     ("word_feats", SentenceWordFeaturesVectorizer()),
        ('tfidf_lemma', Pipeline([
            ("lemma", LemmatizerTransformerPT()),
            ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
        ])),
        ("postag", POSTaggerVectorizerPT())
    ]),
    "es": FeatureUnion([
        ("cv2", CountVectorizer(ngram_range=(1, 2))),
        #  ("word_feats", SentenceWordFeaturesVectorizer()),
        ('tfidf_lemma', Pipeline([
            ("lemma", LemmatizerTransformerES()),
            ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
        ])),
        ("postag", POSTaggerVectorizerES())
    ]),
    "ca": FeatureUnion([
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            #    ("word_feats", SentenceWordFeaturesVectorizer()),
            ('tfidf_lemma', Pipeline([
                ("lemma", LemmatizerTransformerCA()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ])),
            ("postag", POSTaggerVectorizerCA())
        ]),
    "fr": FeatureUnion([
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            ("word_feats", SentenceWordFeaturesVectorizer()),
            ('tfidf_lemma', Pipeline([
                ("lemma", LemmatizerTransformerFR()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ]))
        ]),
    "de": FeatureUnion([
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            ("word_feats", SentenceWordFeaturesVectorizer()),
            ('tfidf_lemma', Pipeline([
                ("lemma", LemmatizerTransformerDE()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ]))
        ]),
    "it": FeatureUnion([
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            ("word_feats", SentenceWordFeaturesVectorizer()),
            ('tfidf_lemma', Pipeline([
                ("lemma", LemmatizerTransformerIT()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ]))
        ]),
    # generic for all languages...
    "default": FeatureUnion([
        ("cv2", CountVectorizer(ngram_range=(1, 2))),
        ("word_feats", SentenceWordFeaturesVectorizer()),
        ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
    ]),
    "naive": FeatureUnion([
        ("word_feats", WordFeaturesVectorizer())
    ]),
    # task specific
    "naive_en": FeatureUnion([
        ("word_feats", SentenceWordFeaturesVectorizer()),
        ("postag", POSTaggerVectorizerEN())
    ])
}
"""


def get_features_pipeline(pipeline_id="default"):
    DEFAULT_PIPELINES = {
        "naive": FeatureUnion([
            ("word_feats", WordFeaturesVectorizer())
        ]),
        "words": FeatureUnion([
            ("word_feats", SingleWordFeaturesVectorizer())
        ]),
        "postag_en": FeatureUnion([
            ("postag", POSTaggerVectorizer(lang="en"))
        ]),
        "pronouns_en": FeatureUnion([
            ("corefiob", PronounTaggerVectorizer(lang="en"))
        ]),
        "coref_en": FeatureUnion([
            ("corefiob", CorefIOBTaggerVectorizer(lang="en"))
        ]),
        "cv2": Pipeline([
            ("cv2", CountVectorizer(ngram_range=(1, 2)))
        ]),
        "lang": Pipeline([
            ("lang", LangFeaturesVectorizer()),
            ("word", SingleWordFeaturesVectorizer())
        ]),
        "tfidf": Pipeline([
            ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
        ]),
        "skipgram2": Pipeline([
            ('skipgram2', SkipGramVectorizer(2, 2))
        ]),
        "tfidf_lemma": Pipeline([
            ("tokenize", TokenizerTransformer()),
            ("lemma", WordNetLemmatizerTransformer()),
            ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
        ]),
        "cv2_lemma": Pipeline([
            ("tokenize", TokenizerTransformer()),
            ("lemma", WordNetLemmatizerTransformer()),
            ("cv2", CountVectorizer(ngram_range=(1, 2)))
        ]),
        # pipelines for question classification
        "questions_en": FeatureUnion([
            ("question_feats", QuestionFeaturesVectorizerEN()),
            ("cv2", CountVectorizer(ngram_range=(1, 2))),
            ('tfidf_lemma', Pipeline([
                ("lemma", WordNetLemmatizerTransformer()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ])),
            ("postag", POSTaggerVectorizer(lang="en"))
        ]),

        # pipelines for OCP classification
        "ocp_kw": Pipeline([
            ("kw", OCPKeywordFeaturesVectorizer())
        ]),
        "ocp_kw_cv2": FeatureUnion([
            ("kw", OCPKeywordFeaturesVectorizer()),
            ("cv2", CountVectorizer(ngram_range=(1, 2)) )
        ]),
        "ocp_kw_tfidf": FeatureUnion([
            ("kw", OCPKeywordFeaturesVectorizer()),
            ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
        ]),
        "ocp_kw_cv2_lemma": FeatureUnion([
            ("kw", OCPKeywordFeaturesVectorizer()),
            ("cv2_lemma", Pipeline([
                ("lemma", WordNetLemmatizerTransformer()),
                ("cv2", CountVectorizer(ngram_range=(1, 2)))
            ])
             )
        ]),
        "ocp_kw_tfidf_lemma": FeatureUnion([
            ("kw", OCPKeywordFeaturesVectorizer()),
            ("tfidf_lemma", Pipeline([
                ("lemma", WordNetLemmatizerTransformer()),
                ('tfidf', TfidfVectorizer(min_df=.05, max_df=.4))
            ])
             )
        ])
    }

    return DEFAULT_PIPELINES[pipeline_id]
