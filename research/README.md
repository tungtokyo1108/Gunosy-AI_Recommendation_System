# Discription 

1. [Data collection](#1-data-collection)
2. [Data pre-precessing](#2-data-pre-processing)
3. [Feature engineering](#3-feature-engineering)
4. [Model training](#4-model-traing)
5. [Model evaluation](#5-model-evaluation)

## Data collection 

[Gunosy_new.py](https://github.com/tungtokyo1108/Gunosy-Data_Analysis/blob/master/research/Gunosy_new.py) and [Gunosy_new notebook]() provides the method to collect data from `https://gunosy.com/`. The data in Gunosy web includes the 8 groups 「エンタメ」、「スポーツ」、「おもしろ」、「国内」、「海外」、「コラム」、「IT・科学」、「グルメ」and each of groups has some sub-groups. However the number of sub-groups among groups are not similar, 「スポーツ」 has the largest number with 1100 news but 「海外」 has only 400 news. Therefore, in order to avoid the imbalanced classification problems and limitation of my computer, the 400 news are sampled randomly for each of 8 groups. The "requests" and "bs4" packages are used to collect the data.  

## Data pre-processing 

