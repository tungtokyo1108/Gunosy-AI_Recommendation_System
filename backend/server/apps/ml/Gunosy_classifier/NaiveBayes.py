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
from gensim.models import word2vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm, tqdm_pandas, tqdm_notebook

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy import interp

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import logsumexp
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")

class NaiveBayes:
    def __init__(self, alpha=0.01):
        path_to_artifacts = "../../research/"
        self.alpha = alpha
        
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
        self.df_pred = df
        
        with open("News_dataset.pickle", "rb") as data:
            self.df_train = pickle.load(data)
            self.df_train = self.df_train.reset_index(drop=True).drop(columns = ["News_length"])

        with open("Updated_news.pickle", "rb") as data:
            self.df_pre_recommend = pickle.load(data)
            self.df_pre_recommend = self.df_pre_recommend.reset_index(drop=True).drop(columns = ["News_length"])
            
        self.df_train["Content_Parsed_1"] = self.df_train["Article"].str.replace("キーワードで気になるニュースを絞りこもう 「いいね」、フォローをしておすすめの記事をチェックしよう。 グノシーについて 公式SNS 関連サイト アプリをダウンロード グノシー | 情報を世界中の人に最適に届ける Copyright © Gunosy Inc. All rights reserved.", '')
        nrows = len(self.df_train)
        wakati_text_list = []
        for row in range(0, nrows):
            text = self.df_train.loc[row]["Content_Parsed_1"]
            wakati_text_list.append(get_wakati_text(text))

        self.df_train["wakati_text"] = wakati_text_list

        self.df_pre_recommend["Content_Parsed_1"] = self.df_pre_recommend["Article"].str.replace("キーワードで気になるニュースを絞りこもう 「いいね」、フォローをしておすすめの記事をチェックしよう。 グノシーについて 公式SNS 関連サイト アプリをダウンロード グノシー | 情報を世界中の人に最適に届ける Copyright © Gunosy Inc. All rights reserved.", '')
        nrows = len(self.df_pre_recommend)
        wakati_text_list = []
        for row in range(0, nrows):
            text = self.df_pre_recommend.loc[row]["Content_Parsed_1"]
            wakati_text_list.append(get_wakati_text(text))

        self.df_pre_recommend["wakati_text"] = wakati_text_list
        
        df = pd.concat([df, self.df_train]).reset_index(drop=True)
        
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
    
    """
    Reference: 
        https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
    """
    
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
             log P^(c) + sum(log P^(t|c))
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

        # The maximum of posteriori (MAP)
        
        self.update_feature_log_distribution(alpha)
        self.update_class_log_distribution()
        jll = self.joint_log_likelihood(X_test)

        predict = self.classes_[np.argmax(jll, axis=1)]

        log_prob_x = logsumexp(jll, axis=1)
        predict_log_prob = jll - np.atleast_2d(log_prob_x).T
        predict_prob = np.exp(predict_log_prob)
        
        return predict_prob, predict

    def recommend(self, input_data):

        if input_data == 0:
            label = "エンタメ"
        elif input_data == 1:
            label = "スポーツ"
        elif input_data == 2:
            label = "グルメ"
        elif input_data == 3:
            label = "海外"
        elif input_data == 4:
            label = "おもしろ"
        elif input_data == 5:
            label = "国内"
        elif input_data == 6:
            label = "IT・科学"
        else:
            label = "コラム"

        self.df_pred["Category"] = label

        df_pred_recommed = self.df_pred[["Category", "Title", "PageLink", "wakati_text"]]
        df_train_recommed = self.df_pre_recommend[["Category", "Title", "PageLink", "wakati_text"]]

        df_recommend = pd.concat([df_pred_recommed, df_train_recommed]).reset_index(drop=True)
        
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
        
        news_recommend = df_recommend[df_recommend.Category == label].reset_index(drop=True)
        #news_inter = news_inter.iloc[[0,1]]
        corpus_recommend = [doc.split() for doc in news_recommend.wakati_text.values]
        model_recommend = word2vec.Word2Vec(corpus_recommend, size=1000, min_count=20, window=10)
        X_recommend = np.zeros((len(news_recommend), model_recommend.wv.vector_size))
        for i, doc in tqdm_notebook(enumerate(news_recommend.wakati_text.values)):
            X_recommend[i, :] = get_doc_mean_vector(doc, model_recommend)

        similar = np.zeros(len(X_recommend))
        for i in range(0, len(X_recommend)):
            #similar.append(cosine_similarity(X_inter[0].reshape(1, -1), X_inter[i].reshape(1, -1)))
            similar[i] = cosine_similarity(X_recommend[0].reshape(1, -1), X_recommend[i].reshape(1, -1))
    
        df_similar = pd.DataFrame(similar, columns=["Cosine_similarity"])
        df_recommendation_unsort = pd.concat([news_recommend[['Title', 'PageLink']], df_similar['Cosine_similarity']], axis=1)
        df_recommendation_unsort = df_recommendation_unsort.sort_values(by=['Cosine_similarity'], ascending=False).reset_index(drop=True)
        self.df_recommendation = df_recommendation_unsort.drop(df_recommendation_unsort.index[0]).reset_index(drop=True)


        return self.df_recommendation
    
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
                "-------------------------------------------------------------------": " ",
                "Recommendation": "The ranked lists With Title, Link, Level of similarity",
                "*********": " ",
                "1st_Title": self.df_recommendation.iloc[0]['Title'],
                "1st_link": self.df_recommendation.iloc[0]['PageLink'],
                "1st_Score_similarity": round(self.df_recommendation.iloc[0]['Cosine_similarity']*100, 2),
                "**********": " ",
                "2nd_Title": self.df_recommendation.iloc[1]['Title'],
                "2nd_link": self.df_recommendation.iloc[1]['PageLink'],
                "2nd_Score_similarity": round(self.df_recommendation.iloc[1]['Cosine_similarity']*100, 2),
                "***********": " ",
                "3rd_Title": self.df_recommendation.iloc[2]['Title'],
                "3rd_link": self.df_recommendation.iloc[2]['PageLink'],
                "3rd_Score_similarity": round(self.df_recommendation.iloc[2]['Cosine_similarity']*100, 2),
                "*************": " ",
                "4th_Title": self.df_recommendation.iloc[3]['Title'],
                "4th_link": self.df_recommendation.iloc[3]['PageLink'],
                "4th_Score_similarity": round(self.df_recommendation.iloc[3]['Cosine_similarity']*100, 2),
                "***************": " ",
                "5th_Title": self.df_recommendation.iloc[4]['Title'],
                "5th_link": self.df_recommendation.iloc[4]['PageLink'],
                "5th_Score_similarity": round(self.df_recommendation.iloc[4]['Cosine_similarity']*100, 2),
                "status: ": "OK"}
        
    def compute_prediction(self, input_links):
        try:
            input_data = self.get_news(input_links)
            X, y, X_pred = self.preprocessing(input_data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            prediction, group = self.estimate_predict(X_train, y_train, X_pred)
            recommendation = self.recommend(group)
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        
        return prediction


# Test 
my_algo = NaiveBayes()
input_links = "https://gunosy.com/articles/RZQor"        
my_algo.compute_prediction(input_links)
        