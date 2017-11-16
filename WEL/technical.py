import sys
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
'''
#text preprocessing
from nltk import word_tokenize
from nltk.corpus import stopwords
'''

'''
stop = set(stopwords.words('english'))
#print (stop)
sentence = "this is a foo bar sentence"

for i in sentence.lower().split():
    if i not in stop:print (i)
'''
'''
A=[[1,2,3],[4,5,6],[7,8,9],[6,6,6]]
print (A)
B=A[0:2]
print (B)
C=A[2:4]
print (C)
'''
tk = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789',num_words=2)
texts = ['I love you.', 'I love you, too.']
tk.fit_on_texts(texts)
tk.texts_to_matrix(texts,mode='tfidf')
#print (tk.shape)
print ("???",tk)
print (texts)
print(tk.word_counts)
#print(a)

print ("end")
os.system('pause')