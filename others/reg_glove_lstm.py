import cus_tools
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
from keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

cus_tools.output()

glove_name="glove.6B.50d.txt"
headlines="Combined_News_DJIA.csv"

maxlen=MAX_SEQUENCE_LENGTH = 200
max_features=MAX_NB_WORDS = 10000
EMBEDDING_DIM = 50
#VALIDATION_SPLIT = 0.2
batch_size = 64
nb_classes = 2

### first, build index mapping words in the embeddings set 
# to their embedding vector
embeddings_index=cus_tools.load_embedding(glove_name)
#print (embeddings_index["the"])

### second, prepare text samples and their labels
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

print ("data shape:",data.shape)
print ("list shape:",(len(headlines),len(headlines[0])))

merged_headlines=[]
for rows in range(len(headlines)):
    temp1=[]
    for cols in range(25):
        temp1=temp1+headlines[rows][cols]
    merged_headlines.append(' '.join(word for word in temp1))


merged_train=merged_headlines[0:1611]
merged_test=merged_headlines[1611:1989]
#merged_test=headlines[1611:1989][0]
tokenizer = Tokenizer(num_words=max_features,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789',lower=False)

tokenizer.fit_on_texts(merged_train)
sequences_train = tokenizer.texts_to_sequences(merged_train)
sequences_test = tokenizer.texts_to_sequences(merged_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)

#y_train = np.array(train["Label"])
#y_test = np.array(test["Label"])

#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

Y=cus_tools.cal_price_rate()[1]#use change
Y_train=Y[0:len(X_train)]
Y_test=np.reshape(Y[len(X_train):],(378,1))



print('X_train shape:', X_train.shape)
#print('X_test shape:', X_test.shape)



print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)
#(20001, 100)

#print ("embedding_matrix[71]:",embedding_matrix[X_train[0][0]])
#print ("embedding_matrix[2629]:",embedding_matrix[X_train[0][-1]])

#print ("embeddings_index.get('georgia'):",embeddings_index.get('georgia'))
#print ("embeddings_index.get('surge'):",embeddings_index.get('surge'))

#print ("embeddings_index.get('the'):",embeddings_index.get('the'))

# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)# Here trainable must be false



#LSTM model
print('Build model...')
model = Sequential()
model.add(embedding_layer)
model.add(SpatialDropout1D(0.2))
#model.add(Dropout(0.2))

model.add(LSTM(EMBEDDING_DIM,dropout=0.2, recurrent_dropout=0.2))#return_sequences=True,
model.add(Dense(128,activation='tanh'))
model.add(Dense(64,activation='tanh'))
model.add(Dense(32,activation='tanh'))


model.add(Dense(1))
model.add(Activation('tanh'))#softmax

model.summary()#print the model

model.compile(loss='mean_squared_error',
              optimizer='adam')
#'binary_crossentropy''mean_squared_error' adam

print('Train...')
#early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=2,
          validation_data=(X_test, Y_test))#,callbacks=[early_stopping]
# make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
test_label=[]
for i in range(len(testPredict)):
    if testPredict[i]>=0:
        test_label.append(1)
    else:
        test_label.append(0)

print (testPredict)
print (test_label)
print (len(test_label))

true_test_label=np.array(test["Label"])

print ("acc:",cus_tools.acc(true_test_label,test_label))

print ("end")
