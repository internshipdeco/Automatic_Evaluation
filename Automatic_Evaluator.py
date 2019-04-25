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


#Loading Training File
df = pd.read_csv("train_dataset.csv", encoding="utf-8")

#Loading Test File
df_test = pd.read_csv("test_dataset.csv")

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer.fit(df[["score_3","score_4","score_5"]])
df[["score_3","score_4","score_5"]] = imputer.transform(df[["score_3","score_4","score_5"]])

df["Essyset"] = df["Essayset"].fillna(method='bfill')

imputer = Imputer(missing_values = "NaN", strategy = "most_frequent", axis = 0)
imputer.fit(df[["clarity","coherent"]])
df[["clarity","coherent"]] = imputer.transform(df[["clarity","coherent"]])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
encoder.fit(df['coherent'].drop_duplicates())
df['coherent'] = encoder.transform(df['coherent'])
#encoder.fit(df['clarity'].drop_duplicates())
#encoder.fit_transform(df['clarity'])
hotencoder = OneHotEncoder(categorical_features = df[["clarity","coherent"]])
df[["clarity","coherent"]] = hotencoder.fit_transform(df[["clarity","coherent"]])

corpus = []
for i in range (0, len(df)):
    review = re.sub('[^a-zA-Z]',' ', df['EssayText'][i])
    review = review.lower()
    review = nltk.word_tokenize(review)
    #ps = PorterStemmer()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

df["Cleaned_text"]=corpus

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
df["final_score"] = df[["score_1","score_2","score_3","score_4","score_5"]].mean(axis=1).astype(int)
y =df["final_score"].values
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X,y)


corpus_test = []
for i in range (0, len(df_test)):
    review = re.sub('[^a-zA-Z]',' ', df['EssayText'][i])
    review = review.lower()
    review = nltk.word_tokenize(review)
    #ps = PorterStemmer()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_test.append(review)

X_test = cv.fit_transform(corpus_test).toarray()
y_pred = classifier.predict(X_test)
df_test["essay_score"] = y_pred
df_resul = df_test[["ID","Essayset","essay_score"]]

df_resul.to_csv('Submission_test.csv', header =[ "id","essay_set","essay_score"] ,index= False)


