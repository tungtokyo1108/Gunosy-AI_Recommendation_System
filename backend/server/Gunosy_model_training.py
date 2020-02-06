# -*- coding: utf-8 -*-
"""
@author: Tung Dang 
"""

import pickle
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from collections import Counter, defaultdict

import MeCab
from gensim.models import word2vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

print("\nStart training model ")

print("\nLoading data training")
with open("News_dataset.pickle", "rb") as data:
    df = pickle.load(data)
df = df.reset_index(drop=True)

print("\nFeature engineering")
df["Content_Parsed_1"] = df["Article"].str.replace("キーワードで気になるニュースを絞りこもう 「いいね」、フォローをしておすすめの記事をチェックしよう。 グノシーについて 公式SNS 関連サイト アプリをダウンロード グノシー | 情報を世界中の人に最適に届ける Copyright © Gunosy Inc. All rights reserved.", '')

###############################################################################

def get_wakati_text(text):
    tagger = MeCab.Tagger("-Owakati")
    wakati_text = tagger.parse(text).strip()
    return wakati_text

###############################################################################
nrows = len(df)
wakati_text_list = []
for row in range(0, nrows):
    
    text = df.loc[row]["Content_Parsed_1"]
    wakati_text_list.append(get_wakati_text(text))

df["wakati_text"] = wakati_text_list

###############################################################################
######################### NaiveBayes-algorithm ################################
###############################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy import interp

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

def get_GridSearchCV_estimator(model, X_train, y_train, X_test, y_test):
    
    alphas = np.logspace(-2,0,20)
    tuned_parameters = [{"alpha": alphas}]
    n_folds = 10
    model = MultinomialNB()
    my_cv = TimeSeriesSplit(n_splits=n_folds).split(X_train)
    gsearch_cv = GridSearchCV(estimator=model, param_grid=tuned_parameters, cv=my_cv, scoring="f1_macro", n_jobs=-1)
    gsearch_cv.fit(X_train, y_train)
    print("\n~~~~~~~~~~~~~~~~~~ The best model ~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Best estimator: ", gsearch_cv.best_estimator_)
    print("Best Score: ", gsearch_cv.best_score_)
    return gsearch_cv

def evaluate_multiclass(best_clf, X_train, y_train, X_test, y_test, 
                        model="Random Forest", num_class=3):
    print("-"*100)
    print("~~~~~~~~~~~~~~~~~~ PERFORMANCE EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Detailed report for the {} algorithm".format(model))
    
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
    points = accuracy_score(y_test, y_pred, normalize=False)
    print("The number of accurate predictions out of {} data points on unseen data is {}".format(
            X_test.shape[0], points))
    print("Accuracy of the {} model on unseen data is {}".format(
            model, np.round(test_accuracy, 2)))
    
    print("Precision of the {} model on unseen data is {}".format(
            model, np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)))
    print("Recall of the {} model on unseen data is {}".format(
           model, np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)))
    print("F1 score of the {} model on unseen data is {}".format(
            model, np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)))
    
    print("\nClassification report for {} model: \n".format(model))
    print(metrics.classification_report(y_test, y_pred))

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("\nThe Confusion Matrix: \n")
    print(cnf_matrix)
    print("\n")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("\n")
    
    return y_pred, y_pred_prob

###############################################################################
##################### NaiveBayes-algorithm for IF-IDF #########################
###############################################################################

# TF-IDF
vectorizer = TfidfVectorizer(use_idf = True, token_pattern=u'(?u)\\b\\w+\\b')
X = vectorizer.fit_transform(df.wakati_text.values)
X = X.toarray()

y = df["Category"].apply(lambda x: 0 
                             if x == "エンタメ" else 1 
                             if x == "スポーツ" else 2
                             if x == "グルメ" else 3
                             if x == "海外" else 4 
                             if x == "おもしろ" else 5
                             if x == "国内" else 6
                             if x == "IT・科学" else 7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nStarting Cross Validation steps...")
gsearch_cv = get_GridSearchCV_estimator("Naive Bayes", X_train, y_train, X_test, y_test)
nb_classifier = gsearch_cv.best_estimator_
nb_classifier.fit(X_train, y_train)
y_pred, y_pred_prob = evaluate_multiclass(nb_classifier, X_train, y_train, X_test, y_test, 
                        model="Naive Bayes", num_class=8)
