#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:05:24 2020

@author: tungutokyo
"""

import joblib
import pickle
import pandas as pd
import numpy as np
import urllib
import requests
import bs4 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import MeCab

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

import warnings
warnings.filterwarnings("ignore")

class NaiveBayes:
    def __init__(self):
        path_to_artifacts = "../../research/"
        
    def get_news(self, link):
        title = []
        thearticle = []

        #print(link)
        paragraphtext = []
        url = link
        page = requests.get(url)
        soup = bs4.BeautifulSoup(page.text, "html.parser")
        atitle = soup.find(class_="article_header_text").find("h1")
        thetitle = atitle.get_text()
    
        articletext = soup.find_all("p")
        for paragraph in articletext:
            text = paragraph.get_text()
            paragraphtext.append(text)
    
        title.append(thetitle)
        thearticle.append(paragraphtext)

        myarticle = [" ".join(article) for article in thearticle]

        data = {
            "Title": title,
            "Article": myarticle,
            "PageLink": link}

        news = pd.DataFrame(data=data)
        cols = ["Title", "Article", "PageLink"]
        news = news[cols]
    
        return news
        
    def preprocessing(self, input_data):
        
        df = input_data.reset_index(drop=True)
        df["Content_Parsed_1"] = df["Article"].str.replace("キーワードで気になるニュースを絞りこもう 「いいね」、フォローをしておすすめの記事をチェックしよう。 グノシーについて 公式SNS 関連サイト アプリをダウンロード グノシー | 情報を世界中の人に最適に届ける Copyright © Gunosy Inc. All rights reserved.", '')
        
        def get_wakati_text(text):
            tagger = MeCab.Tagger("-Owakati")
            wakati_text = tagger.parse(text).strip()
            return wakati_text
        
        nrows = len(df)
        wakati_text_list = []
        for row in range(0, nrows):
    
            text = df.loc[row]["Content_Parsed_1"]
            wakati_text_list.append(get_wakati_text(text))

        df["wakati_text"] = wakati_text_list
        
        with open("News_dataset.pickle", "rb") as data:
            df_train = pickle.load(data)
            df_train = df_train.reset_index(drop=True).drop(columns = ["News_length"])

        df_train["Content_Parsed_1"] = df_train["Article"].str.replace("キーワードで気になるニュースを絞りこもう 「いいね」、フォローをしておすすめの記事をチェックしよう。 グノシーについて 公式SNS 関連サイト アプリをダウンロード グノシー | 情報を世界中の人に最適に届ける Copyright © Gunosy Inc. All rights reserved.", '')
        nrows = len(df_train)
        wakati_text_list = []
        for row in range(0, nrows):
            text = df_train.loc[row]["Content_Parsed_1"]
            wakati_text_list.append(get_wakati_text(text))

        df_train["wakati_text"] = wakati_text_list
        
        df = pd.concat([df, df_train]).reset_index(drop=True)
        
        vectorizer = TfidfVectorizer(use_idf = True, token_pattern=u'(?u)\\b\\w+\\b')
        X = vectorizer.fit_transform(df.wakati_text.values)
        X = X.toarray()
        X_pred = X[0].reshape(1,-1)
        X = np.delete(X, 0, axis=0)
        
        df = df.drop(df.index[0])
        y = df["Category"].apply(lambda x: 0 
                             if x == "エンタメ" else 1 
                             if x == "スポーツ" else 2
                             if x == "グルメ" else 3
                             if x == "海外" else 4 
                             if x == "おもしろ" else 5
                             if x == "国内" else 6
                             if x == "IT・科学" else 7)
        
        return X, y, X_pred
    
    def predict(self, X, y, X_pred):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        #alphas = np.logspace(-2,0,50)
        #tuned_parameters = [{"alpha": alphas}]
        #n_folds = 10
        #model = MultinomialNB()
        #my_cv = TimeSeriesSplit(n_splits=n_folds).split(X_train)
        #gsearch_cv = GridSearchCV(estimator=model, param_grid=tuned_parameters, cv=my_cv, scoring="f1_macro", n_jobs=-1)
        #gsearch_cv.fit(X_train, y_train)
        #nb_classifier = gsearch_cv.best_estimator_
        nb_classifier = MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
        nb_classifier.fit(X_train, y_train)
        y_pred = nb_classifier.predict(X_pred)
        y_pred_prob = nb_classifier.predict_proba(X_pred)
        return y_pred_prob

    def postprocessing(self, input_data):
        
        data_pred = {'label': ['エンタメ', 'スポーツ', 'グルメ', '海外', 'おもしろ', '国内', 'IT・科学', 'コラム'],
                     'prob': [input_data.item(0,0), input_data.item(0,1), input_data.item(0,2), 
                              input_data.item(0,3), input_data.item(0,4), input_data.item(0,5),
                              input_data.item(0,6), input_data.item(0,7)]}
                     
        data_pred = pd.DataFrame(data=data_pred)
        data_pred = data_pred.sort_values(by=['prob'], ascending=False).reset_index(drop=True)
            
        return {"Group_1st: " : data_pred.loc[0, 'label'],
                "Probablity for Group_1st is: ": round(data_pred.loc[0, 'prob']*100,2),
                "Group_2nd: ": data_pred.loc[1, 'label'],
                "Probablity for Group_2nd is: ": round(data_pred.loc[1, 'prob']*100,2),
                "status: ": "OK"}
        
    def compute_prediction(self, input_links):
        try:
            input_data = self.get_news(input_links)
            X, y, X_pred = self.preprocessing(input_data)
            prediction = self.predict(X, y, X_pred)
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        
        return prediction 


# Test 
my_algo = NaiveBayes()
input_links = "https://gunosy.com/articles/RNj4e"        
my_algo.compute_prediction(input_links)
        
      
        
        
    