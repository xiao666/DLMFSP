import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#text preprocessing
from nltk import word_tokenize
from nltk.corpus import stopwords

import sys
import os

stopwords = set(stopwords.words('english'))

data = pd.read_csv('Combined_News_DJIA.csv')

example=data.iloc[0,0]
print(example)