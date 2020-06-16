"""
Utility Code for Rakuten Takehome

Emma Nechamkin
June 14, 2020

Includes helper functions to streamline analysis
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import (
    TfidfTransformer,
    TfidfVectorizer,
    CountVectorizer,
)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    f1_score,
)

from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def get_category_levels(df, cat_col="CategoryIdPath", divider=">"):
    """
    Modifies df in place to get a prediction label, separates string into 
    each category as new 
    
    Inputs:
        df (dataframe)
        cat_col (str) name of what to split 
        divider (str) what to split text on 
    
    Returns: None, modifies df in place
    """
    max_num_cat = df[cat_col].str.count(divider).max() + 1
    df[[f"category_level_{x}" for x in range(max_num_cat)]] = (
        df[cat_col].str.split(divider, expand=True).apply(pd.to_numeric)
    )


def load_standard_x_y(df, x="Title", y="category_level_0"):
    get_category_levels(df)
    return df["Title"].to_list(), df["category_level_0"]


def evaluate_model(y_true, y_pred):
    """
    Wrapper for some standard evaluation metrics
    """
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted')}")
    print(f"F1: {f1_score(y_true, y_pred,  average='weighted')}")


class CountVectorizerWithStemming(CountVectorizer):
    """
    Custom instance of CountVectorizer to include Stemming, since SKLearn does not 
    by default. 
    """

    def build_analyzer(self):
        analyzer = super(CountVectorizerWithStemming, self).build_analyzer()
        stemmer = PorterStemmer()
        return lambda text_field: (stemmer.stem(word) for word in analyzer(text_field))


def create_pipeline(
    preprocess_step,
    classifier,
    param_grid={"alpha": [0.25, 1, 1.75], "fit_prior": [True, False]},
    max_features=12000,
    scoring="balanced_accuracy",
    stop_words=stopwords.words("english"),
):
    """
    Helper function to simplify creating different MNB classifiers
    
    Inputs:
        preprocess_step (str) one of bow, bow_stem, tfidf. How features will be generated
        param_grid (dict) MNB parameters to try
        max_features (int) how many features to include
        scoring (str) scoring type for gridCV
        stop_words (list or stopwords dictionary) the stop words to exclude
    
    Returns sklearn pipeline object
    """
    assert preprocess_step in [
        "bow",
        "bow_stem",
        "tfidf",
    ], "preprocess_step should be one of bow, bow_stem, or tfidf"

    if preprocess_step == "bow":
        preprocessor = CountVectorizer(stop_words=stop_words, max_features=max_features)
    elif preprocess_step == "bow_stem":
        preprocessor = CountVectorizerWithStemming(
            stop_words=stop_words, max_features=max_features
        )
    else:
        preprocessor = TfidfVectorizer(stop_words=stop_words, max_features=max_features)

    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "classifier",
                GridSearchCV(
                    classifier,
                    param_grid=param_grid,
                    cv=5,
                    n_jobs=-1,
                    scoring="balanced_accuracy",
                ),
            ),
        ]
    )


def fit_predict_score_pipeline(pipe, X, y):
    """
    Helper function to reduce code -- fits, predicts, and scores
    classifiers. 
    
    Inputs:
        pipe (sklearn pipeline)
        X (array) X data
        y (array or series) target data
        
    Returns: fit clf, prints in place
    """
    clf = pipe.fit(X, y)
    pred_y = clf.predict(X)
    evaluate_model(y, pred_y)
    return clf


def show_balance_table(
    df,
    colname,
    clf=True,
    classifier_pipe=None,
    train_X=None,
    y_data=None,
    target="category_level_0",
):
    """
    Produce a table that shows how accurate each category was
    
    Inputs:
        df (dataframe) 
        colname (str) colname to create
        clf (bool) if True, run predictions. If False, use y_data
        classifier_pipe (pipe) sklearn fitted pipe for use if 
            clf is true -- will predict y data
        train_X (array) X data if clf is True
        y_data (optional, vector) the y_data, if clf=False
        target (str) colname for target
    
    Returns: none, displays table. Modifies frame in place.
    """
    if clf:
        y_data = classifier_pipe.predict(train_X)

    df[colname] = y_data == df[target] * 1.0
    df["prediction"] = y_data
    print(
        df.groupby([target])[colname]
        .value_counts(normalize=True)
        .unstack()
        .sort_values(by=True)
    )