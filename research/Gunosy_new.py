# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:15:39 2020

@author: dangt
"""

import time 
import urllib
import requests 
import bs4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 

###############################################################################

def extract_inform(url, name_sec):
    page = requests.get(url)

    soup = bs4.BeautifulSoup(page.content, "html.parser")
    weblinks = soup.find_all("div", class_="list_content")

    jp_pagelinks = []
    for link in weblinks:
        url = link.contents[0].find_all("a")[0]
        jp_pagelinks.append(url.get("href"))

    title = []
    thearticle = []

    for link in jp_pagelinks:
        print(link)
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

    data = {"Category": name_sec,
            "Title": title,
            "Article": myarticle,
            "PageLink": jp_pagelinks}

    news = pd.DataFrame(data=data)
    cols = ["Category", "Title", "Article", "PageLink"]
    news = news[cols]
    
    return news 
    

###############################################################################


url = "https://gunosy.com/"

page = requests.get(url)
html = page.text
soup = bs4.BeautifulSoup(html, "html.parser")

ul = soup.find("nav", class_="nav").ul

section_list = []
title = []
for li in ul.find_all("li"):
    title.append(li.a.get_text())
    section_list.append(li.a.get("href"))

dic_complete = dict()

for i in range(1,3):
    class_name = "nav_color_" + str(i)
    name_sec = soup.find("li", class_=class_name).a.get_text()
    sec = soup.find("li", class_=class_name).ul
    sec_list = []
    tit_list = []
    for li in sec.find_all("li"):
        tit_list.append(li.a.get_text())
        sec_list.append(li.a.get("href"))
        
    add_part = ["?page=1", "?page=2", "?page=3", "?page=4", "?page=5"]
    sec_list_complete = []
    for url in sec_list:
        for add in add_part:
            sec_list_complete.append(url + add)
    
    dic = dict()
    for page_url in sec_list_complete:
        dic[page_url] = extract_inform(page_url, name_sec)
        
    news_sec = pd.concat(dic.values(), ignore_index = True)
    
    dic_complete[name_sec] = news_sec
    
news = pd.concat(dic_complete.values(), ignore_index = True)

###############################################################################

news_enter = pd.read_csv("News_entertainment.csv")
news_sport = pd.read_csv("News_sport.csv")
news_inter = pd.read_csv("News_interting.csv")
news_domestic = pd.read_csv("News_domestic.csv")
news_oversea = pd.read_csv("News_oversea.csv")
news_column = pd.read_csv("News_column.csv")
news_IT = pd.read_csv("News_IT.csv")
news_gourmet = pd.read_csv("News_gourmet.csv")

news_enter_sample = news_enter.sample(n = 200).reset_index(drop=True)
news_sport_sample = news_sport.sample(n = 200).reset_index(drop=True)
news_inter_sample = news_inter.sample(n = 200).reset_index(drop=True)
news_domestic_sample = news_domestic.sample(n = 200).reset_index(drop=True)
news_column_sample = news_column.sample(n = 200).reset_index(drop=True)
news_IT_sample = news_IT.sample(n = 200).reset_index(drop=True)
news_gourmet_sample = news_gourmet.sample(n = 200).reset_index(drop=True)

news = pd.concat([news_enter_sample, news_sport_sample, news_inter_sample, 
                  news_domestic_sample, news_oversea, news_column_sample, 
                  news_IT_sample, news_gourmet_sample], ignore_index = True)

news_shuffle = shuffle(news).reset_index(drop=True)
news_shuffle.to_csv("News.csv", index = False)


































