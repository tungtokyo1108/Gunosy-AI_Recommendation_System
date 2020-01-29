#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:51:24 2020

@author: tungutokyo
"""

import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from collections import Counter, defaultdict

import MeCab
from gensim.models import word2vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm, tqdm_pandas, tqdm_notebook

with open("News_dataset.pickle", "rb") as data:
    df = pickle.load(data)
df = df.reset_index(drop=True)

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
######################### RandomForest-algorithm ##############################
###############################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy import interp

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

def get_RandSearchCV(X_train, y_train, X_test, y_test, scoring):
    from sklearn.model_selection import TimeSeriesSplit
    from datetime import datetime as dt 
    st_t = dt.now()
    # Numer of trees are used
    n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]
    #n_estimators = list(np.arange(100,1000,50))
    #n_estimators = [1000]
    
    # Maximum depth of each tree
    max_depth = [5, 10, 25, 50, 75, 100]
    
    # Minimum number of samples per leaf 
    min_samples_leaf = [1, 2, 4, 8, 10]
    
    # Minimum number of samples to split a node
    min_samples_split = [2, 4, 6, 8, 10]
    
    # Maximum numeber of features to consider for making splits
    max_features = ["auto", "sqrt", "log2", None]
    
    hyperparameter = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf,
                      'min_samples_split': min_samples_split,
                      'max_features': max_features}
    
    cv_timeSeries = TimeSeriesSplit(n_splits=5).split(X_train)
    base_model_rf = RandomForestClassifier(criterion="gini", random_state=42)
    
    # Run randomzed search 
    n_iter_search = 30
    
    rsearch_cv = RandomizedSearchCV(estimator=base_model_rf, 
                                   random_state=42,
                                   param_distributions=hyperparameter,
                                   n_iter=n_iter_search,
                                   cv=cv_timeSeries,
                                   scoring=scoring,
                                   n_jobs=-1)
    
    rsearch_cv.fit(X_train, y_train)
    #f = open("output.txt", "a")
    print("Best estimator obtained from CV data: \n", rsearch_cv.best_estimator_)
    print("Best Score: ", rsearch_cv.best_score_)
    return rsearch_cv

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
    
    plt.figure(figsize=(15,15))
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("\nThe Confusion Matrix: \n")
    print(cnf_matrix)
    
    cmap = plt.cm.Blues
    sns.heatmap(cnf_matrix_norm, annot=True, cmap=cmap, fmt=".2f", annot_kws={"size":15})
    plt.title("The Normalized Confusion Matrix", fontsize=20)
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    plt.show()
    
    print("\nROC curve and AUC")
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)
    y_test_cat = np.array(pd.get_dummies(y_test))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_class):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_cat[:,i], y_pred_prob[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(15,15))
    plt.plot(fpr["macro"], tpr["macro"], 
         label = "macro-average ROC curve with AUC = {} - Accuracy = {}%".format(
                 round(roc_auc["macro"], 2), round(test_accuracy, 2)),
         color = "navy", linestyle=":", linewidth=4)
    #colors = cycle(["red", "orange", "blue", "pink", "green"])
    colors = sns.color_palette()
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label = "ROC curve of class {0} (AUC = {1:0.2f})".format(i, roc_auc[i]))   
    plt.plot([0,1], [0,1], "k--", lw=3, color='red')
    plt.title("ROC-AUC for {} model".format(model), fontsize=20)
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.legend(loc="lower right")
    plt.show()
    
    return y_pred, y_pred_prob

###############################################################################
##################### RandomForest-algorithm for IF-IDF #########################
###############################################################################

# BoW
vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
X = vectorizer.fit_transform(df.wakati_text.values)
X = X.toarray()

# TF-IDF
vectorizer = TfidfVectorizer(use_idf = True, token_pattern=u'(?u)\\b\\w+\\b')
X = vectorizer.fit_transform(df.wakati_text.values)
X = X.toarray()

# Word2Vec 
corpus = [doc.split() for doc in df.wakati_text.values]
model_w2v = word2vec.Word2Vec(corpus, size=10000, min_count=20, window=10)

def get_doc_swem_max_vector(doc, model):
    words = doc.split()
    word_cnt = 0
    vector_size = model.vector_size
    doc_vector = np.zeros((len(words), vector_size))
    
    for i, word in enumerate(words):
        try:
            word_vector = model.wv[word]
        except KeyError:
            word_vector = np.zeros(vector_size)
        
        doc_vector[i, :] = word_vector
    
    doc_vector = np.max(doc_vector, axis=0)
    return doc_vector

X = np.zeros((len(df), model_w2v.wv.vector_size))

for i, doc in tqdm_notebook(enumerate(df.wakati_text.values)):
    X[i, :] = get_doc_swem_max_vector(doc, model_w2v)


def get_doc_mean_vector(doc, model):
    doc_vector = np.zeros(model.vector_size)
    words = doc.split()
    word_cnt = 0 
    for word in words:
        try:
            word_vector = model.wv[word]
            doc_vector += word_vector 
            word_cnt += 1
        except KeyError:
            pass
    doc_vector /= word_cnt 
    return doc_vector

X = np.zeros((len(df), model_w2v.wv.vector_size))

for i, doc in tqdm_notebook(enumerate(df.wakati_text.values)):
    X[i, :] = get_doc_mean_vector(doc, model_w2v)


# Doc2Vec
corpus = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(df.wakati_text.values)]
model = Doc2Vec(vector_size=1000)
model.build_vocab(corpus)
model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

X = np.array([model.infer_vector(doc.split()) for doc in df.wakati_text.values])

###############################################################################

y = df["Category"].apply(lambda x: 0 
                             if x == "エンタメ" else 1 
                             if x == "スポーツ" else 2
                             if x == "グルメ" else 3
                             if x == "海外" else 4 
                             if x == "おもしろ" else 5
                             if x == "国内" else 6
                             if x == "IT・科学" else 7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_vectors, X_test_vectors = standardize(X_train, X_test)

print("Starting Cross Validation steps...")
rsearch_cv = get_RandSearchCV(X_train, y_train, X_test, y_test, "f1_macro")

random_forest = rsearch_cv.best_estimator_
random_forest.fit(X_train, y_train)
y_pred, y_pred_prob = evaluate_multiclass(random_forest, X_train, y_train, X_test, y_test, 
                        model="Random Forest", num_class=8)











































