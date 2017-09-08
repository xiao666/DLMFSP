from glove import Glove, Corpus
import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
print (sys.version)#3.6


###load corpus words
dict=[]
#dict=Glove.load_stanford("test.txt")
dict=Glove.load_stanford("glove.6B.50d.txt")


corpus_words=dict.dictionary.keys()
print ("corpus load end...")


###load sample paragraph
sample = pd.read_csv('Combined_News_DJIA.csv')


###transfer the sampple paragraph




