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
#print(corpus_words)

'''
#simple exp
if ",," in corpus_words:
    print("yes")
else:
    print ("no")

'''

###load dataset words
"""
data = pd.read_csv('Combined_News_DJIA.csv')
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
"""
train=pd.read_csv('Combined_News_DJIA.csv')
trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
print(basictrain.shape)

data_words=basicvectorizer.get_feature_names()
#print(data_words[5])

###check coverage rate
temp=0
for i in range(len(data_words)):
    if data_words[i] in corpus_words:
        temp=temp+1

print (temp)
print (temp/len(data_words))

print ("end")
os.system('pause')