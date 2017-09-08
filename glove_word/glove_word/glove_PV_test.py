from glove import Glove, Corpus
import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
print (sys.version)#3.6

'''
###load corpus words
dict=[]
#dict=Glove.load_stanford("test.txt")
dict=Glove.load_stanford("glove.6B.50d.txt")


corpus_words=dict.dictionary.keys()
print ("corpus load end...")
'''

###load sample paragraph
train=pd.read_csv('test_data.csv')
trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))

print (trainheadlines[0])

basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
#print(basictrain.shape)


data_words=basicvectorizer.get_feature_names()
#print (data_words)
###transfer the sampple paragraph



###end
print ("end")
os.system('pause')


