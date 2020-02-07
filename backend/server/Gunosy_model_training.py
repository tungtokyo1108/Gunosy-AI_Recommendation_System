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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import logsumexp
import warnings
warnings.filterwarnings("ignore")

class NaiveBayes:
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def count(self, X, Y):
        """Count and smooth feature occurrences.
           feature_count_: the number of occurances of term in training documents from class
           class_count_: the number of classes
        """
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def update_feature_log_distribution(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities
            Equation 119: 
            log P^(t|c) = log(T_ct + alpha) - log (sum(T_ct' + alpha))
        """
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))

    def joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X
            Equation 115:
             log P^(c) + log P^(t|c)
        """
        return (safe_sparse_dot(X, self.feature_log_prob_.T) +
                self.class_log_prior_)
        
    def update_class_log_distribution(self):
        """ Equation 116:
                log P^(c) = log(Nc) - log(N)
            Nc: the number of documents in class c 
            N: the total number of documents 
        """
        n_classes = len(self.classes_)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            log_class_count = np.log(self.class_count_)

        # empirical prior, with sample_weight taken into account
        self.class_log_prior_ = (log_class_count -
                                     np.log(self.class_count_.sum()))
            
    def starting_values(self, n_effective_classes, n_features):
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                       dtype=np.float64)
        
    def estimate_predict(self, X, y, X_test):
    
        _, n_features = X.shape
        self.n_features_ = n_features

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        n_effective_classes = Y.shape[1]

        self.starting_values(n_effective_classes, n_features)
        self.count(X, Y)
        alpha = 0.01
        self.update_feature_log_distribution(alpha)
        self.update_class_log_distribution()
        # The maxium of posteriori (MAP)
        jll = self.joint_log_likelihood(X_test)
        log_prob_x = logsumexp(jll, axis=1)
        predict_log_prob = jll - np.atleast_2d(log_prob_x).T
        predict_prob = np.exp(predict_log_prob)

        predict = self.classes_[np.argmax(jll, axis=1)]
        
        return predict, predict_prob

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

def evaluate_multiclass(X_train, y_train, X_test, y_test, 
                        model="Random Forest", num_class=3):
    print("-"*100)
    print("~~~~~~~~~~~~~~~~~~ PERFORMANCE EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Detailed report for the {} algorithm".format(model))
    
    #best_clf.fit(X_train, y_train)
    #y_pred = best_clf.predict(X_test)
    #y_pred_prob = best_clf.predict_proba(X_test)

    nb = NaiveBayes()
    y_pred, y_pred_prob = nb.estimate_predict(X_train, y_train, X_test)

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

def nb_cv_roc(X, y, num_cv = 5, random_state=None):
    
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    
    if isinstance(y, list):
        y = np.asarray(y)
    
    alphas = np.logspace(-2,0,20)
    
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    
    for alpha in alphas:
        print("\nThe value of alpha: ", alpha)
        nb = NaiveBayes(alpha = alpha)
        test_accuracies = 0
        test_precisions = 0
        test_recalls = 0
        test_f1s = 0
        cv_count = 0
        for train, test in skf.split(X,y):
            y_pred, y_pred_pro = nb.estimate_predict(X.iloc[train], y.iloc[train], X.iloc[test])
            test_accuracy = metrics.accuracy_score(y.iloc[test], y_pred, normalize = True) * 100
            test_accuracies += test_accuracy
            test_precision = metrics.precision_score(y.iloc[test], y_pred, average="macro")
            test_precisions += test_precision
            test_recall_score = metrics.recall_score(y.iloc[test], y_pred, average="macro")
            test_recalls += test_recall_score
            test_f1_score = metrics.f1_score(y.iloc[test], y_pred, average="macro")
            test_f1s += test_f1_score
            cv_count += 1
        
        test_accuracies /= cv_count
        test_precisions /= cv_count
        test_recalls /= cv_count
        test_f1s /= cv_count

        print ({i: j for i, j in 
            zip(("Accuracy", "Precision_Score", "Recall_Score", "F1_Score"),
                (test_accuracies, test_precisions, test_recalls, test_f1s))})

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

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nStarting Cross Validation steps...")
#gsearch_cv = get_GridSearchCV_estimator("Naive Bayes", X_train, y_train, X_test, y_test)
#nb_classifier = gsearch_cv.best_estimator_
#nb_classifier.fit(X_train, y_train)
#y_pred, y_pred_prob = evaluate_multiclass(X_train, y_train, X_test, y_test, 
#                        model="Naive Bayes", num_class=8)

nb_cv_roc(X, y, num_cv = 5)