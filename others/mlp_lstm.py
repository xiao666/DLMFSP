import nltk
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

import sys
import os

from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer


data = pd.read_csv('../Dataset/Combined_News_DJIA.csv')
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))

#basicvectorizer = CountVectorizer()
#basictrain = basicvectorizer.fit_transform(trainheadlines)
#print(basictrain.shape)


batch_size = 32
nb_classes = 2
advancedvectorizer = TfidfVectorizer( max_features = 31675)
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advancedtest = advancedvectorizer.transform(testheadlines)
print(advancedtrain.shape)

X_train = advancedtrain.toarray()
X_test = advancedtest.toarray()

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(train["Label"])
y_test = np.array(test["Label"])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.mean(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]

# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(512, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Training...")
model.fit(X_train, Y_train, epochs=2, batch_size=32, validation_split=0.15)

print("Generating test predictions...")
preds14 = model.predict_classes(X_test, verbose=0)
acc14 = accuracy_score(test["Label"], preds14)

print('prediction accuracy: ', acc14)