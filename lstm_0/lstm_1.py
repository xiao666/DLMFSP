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

#text preprocessing
from nltk import word_tokenize
from nltk.corpus import stopwords

#import keras.backend as K #calculate gradient
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#lstm

#remove stop_words
stop = set(stopwords.words('english'))
#print (stop)
sentence = "this is a foo bar sentence"


for i in sentence.lower().split():
    if i not in stop:print (i)



#data import
data = pd.read_csv('Combined_News_DJIA.csv')

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

#train = data[data['Date'] < '2015-06-01']
#test = data[data['Date'] > '2015-6-01']

#date process
trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
#basicvectorizer = CountVectorizer()
#basictrain = basicvectorizer.fit_transform(trainheadlines)
#print(basictrain.shape)
#(1611 days,31675 words)
print(len(trainheadlines),len(trainheadlines[0]))

testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))



'''
#lstm
max_features = 10000
maxlen = 200
batch_size = 32
nb_classes = 2

# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(trainheadlines)
sequences_train = tokenizer.texts_to_sequences(trainheadlines)
sequences_test = tokenizer.texts_to_sequences(testheadlines)

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)


y_train = np.array(train["Label"])
y_test = np.array(test["Label"])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

'''
'''
#modeling

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))

model.add(SpatialDropout1D(0.2))
#model.add(Dropout(0.5))
model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2))
#,                dropout=0.5, recurrent_dropout=0.5)
model.add(Dense(nb_classes))
model.add(Activation('softmax'))#softmax

model.summary()#print the model

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#'binary_crossentropy''mean_squared_error' adam

print('Train...')
model.fit(X_train, Y_train, batch_size=batch_size, epochs=3,
          validation_data=(X_test, Y_test))
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)
acc = accuracy_score(test['Label'], preds)

print('prediction accuracy: ', acc)

'''
print ("end")

os.system('pause')