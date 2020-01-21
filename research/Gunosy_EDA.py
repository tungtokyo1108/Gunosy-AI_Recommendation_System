# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:40:50 2020

@author: dangt
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")

###############################################################################

df_news = pd.read_csv("News.csv")

df_news["News_length"] = df_news["Article"].str.len()
plt.figure(figsize=(12,6))
sns.distplot(df_news["News_length"]).set_title("News_length_distribution")

df_news["News_length"].describe()

quantile_95 = df_news["News_length"].quantile(0.95)
df_news_95 = df_news[df_news["News_length"] < quantile_95]
plt.figure(figsize=(12, 6))
sns.distplot(df_news_95["News_length"]).set_title("News length distribution")

sns.set(font=["IPAMincho"])
plt.figure(figsize=(12,6))
sns.boxplot(data=df_news, x = "Category", y = "News_length")

sns.set(font=["IPAMincho"])
plt.figure(figsize=(12,6))
sns.boxplot(data=df_news_95, x = "Category", y = "News_length")

with open("News_dataset.pickle", "wb") as output:
    pickle.dump(df_news_95, output)