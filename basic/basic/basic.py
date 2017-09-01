import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SpatialDropout1D
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from sklearn.metrics import accuracy_score
from keras import optimizers

import keras.backend as K #calculate gradient
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



#import data
data = pd.read_csv('Combined_News_DJIA.csv')

#data = data[data['Date'] < '2015-01-01']

headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,2:27]))


#parameters
max_features = 10000
maxlen = 200
batch_size = 32
nb_classes = 2

# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(headlines)
sequences = tokenizer.texts_to_sequences(headlines)
print('Pad sequences (samples x time)')
X = sequence.pad_sequences(sequences, maxlen=maxlen)
Y = np_utils.to_categorical(data["Label"], nb_classes)

print ('X shape',X.shape)

#LSTM model
model = Sequential()
model.add(Embedding(max_features, output_dim=128))
model.add(Dropout(0.5))
model.add(LSTM(128,dropout=0.5,recurrent_dropout=0.5))
model.add(Dense(nb_classes, activation='softmax'))
model.summary()#print the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X,Y,validation_split=0.2, batch_size=32, epochs=3)