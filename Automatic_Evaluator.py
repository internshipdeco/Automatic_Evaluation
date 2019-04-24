# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:10:51 2019

@author: SHRIKRISHNA
"""

import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

#Loading Training File
df = pd.read_csv("train_dataset.csv")

#Loading Test File
df_test = pd.read_csv("test_dataset.csv")


corpus = []
for i in range (0, len(df)):
    review = re.sub('[^a-zA-Z]',' ', df['EssayText'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    

y = df[["score_1","score_2","score_3","score_4","score_5"]].mean(axis=1)