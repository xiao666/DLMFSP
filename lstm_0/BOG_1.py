#without text preprocessing techniques

from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
import numpy as np
import os
import pandas as pd
from collections import Counter
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import SpatialDropout1D
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from sklearn.metrics import accuracy_score
from keras import optimizers
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


###LSTM###
#import data
data = pd.read_csv('Combined_News_DJIA.csv')
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

headlines=[[0 for x in range(25)] for y in range(len(data.index))]#len(data.index)

stopwords = set(stopwords.words('english'))#+list(punctuation)
print ("num of stop words:",len(stopwords))#
#print (stopwords)

for row in range(len(data.index)):#len(data.index)
    for col in range(25):
        temp0=str(data.iloc[row,(col+2)])
        temp0=temp0.lower()
        temp=HashingVectorizer().build_tokenizer()(temp0)
        #=========code below remove stopwords==================
        #temp=[s for s in temp if s not in stopwords]
        headlines[row][col]=temp

#example headlines:
#"b""Georgia 'downs two Russian warplanes' as countries move to brink of war"""
#b"The commander of a Navy air reconnaissance squadron that provides the President 
#and the defense secretary the airborne ability to command the nation's nuclear weapons has been relieved of duty"

print ("data shape:",data.shape)
print ("list shape:",(len(headlines),len(headlines[0])))

print (headlines[0][0])
print (headlines[0][24])

merged_headlines=[]
for rows in range(len(headlines)):
    temp1=[]
    for cols in range(25):
        temp1=temp1+headlines[rows][cols]
    merged_headlines.append(' '.join(word for word in temp1))

print (len(merged_headlines),len(merged_headlines[0]),len(merged_headlines[1]))#num_days,length of N0.0 string, length of NO.1 string
#1989,2181,1526
print(merged_headlines[0])

merged_train=merged_headlines[0:1611]
merged_test=merged_headlines[1611:1989]

#basicvectorizer = CountVectorizer()
#basictrain = basicvectorizer.fit_transform(merged_train)
#print(basictrain.shape)
#not removed:(1611,31675)

max_features = 10000   #size of vocabulary 10000
maxlen = 200 #max length of sequence
#cut texts after this number of words (among top max_features most common words)
batch_size = 32 
nb_classes = 2
#embeddeing_dims =  #dimensions of word vector

# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=max_features,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789')

tokenizer.fit_on_texts(merged_train)

print ("most occurence words:")
most_common_word=Counter(tokenizer.word_counts)
print (most_common_word.most_common(50))


sequences_train = tokenizer.texts_to_sequences(merged_train)
sequences_test = tokenizer.texts_to_sequences(merged_test)

#pad sequences to fixed length
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)

y_train = np.array(train["Label"])
y_test = np.array(test["Label"])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)



#'''
#LSTM model
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2))
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
#'''

print ("end")
os.system('pause')
