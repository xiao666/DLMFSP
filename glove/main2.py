import cus_tools
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer

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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

cus_tools.output()

glove_name="glove.6B.50d.txt"
headlines="Combined_News_DJIA.csv"

maxlen=MAX_SEQUENCE_LENGTH = 200
max_features=MAX_NB_WORDS = 10000
EMBEDDING_DIM = 50
#VALIDATION_SPLIT = 0.2
batch_size = 32
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
separeted_headlines=[[0 for x in range(25)] for y in range(len(data.index))]#len(data.index)

stopwords = set(stopwords.words('english'))#+list(punctuation)
print ("num of stop words:",len(stopwords))#153
#print (stopwords)

for row in range(len(data.index)):#len(data.index)
    for col in range(25):
        temp0=str(data.iloc[row,(col+2)])
        temp0=temp0.lower()
        temp=HashingVectorizer().build_tokenizer()(temp0)
        #=========code below remove stopwords==================
        #temp=[s for s in temp if s not in stopwords]
        headlines[row][col]=temp
        separeted_headlines[row][col]=' '.join(word for word in temp)

print ("data shape:",data.shape)
print ("list shape:",(len(headlines),len(headlines[0])))
print (headlines[0][0])
print (separeted_headlines[0][0])

merged_headlines=[]
for rows in range(len(headlines)):
    temp1=[]
    for cols in range(25):
        temp1=temp1+headlines[rows][cols]
    merged_headlines.append(' '.join(word for word in temp1))

merged_train=merged_headlines[0:1611]
#merged_test=merged_headlines[1611:1989]
print (type(merged_train))

'''
#single test_headlines
get 25 separeted predictions
'''
sep_test_headlines_1=[0 for x in range(25)]
for k in range(25):
    temp=[row[k] for row in separeted_headlines]
    temp=temp[1611:1989]
    sep_test_headlines_1[k]=temp
print ("len(sep_test_headlines_1)",len(sep_test_headlines_1))

print ("exp separate 1 ")

#==============================================
def prediction(merged_train,merged_test):

    tokenizer = Tokenizer(num_words=max_features,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789',lower=False)

    tokenizer.fit_on_texts(merged_train)
    sequences_train = tokenizer.texts_to_sequences(merged_train)
    sequences_test = tokenizer.texts_to_sequences(merged_test)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)

    y_train = np.array(train["Label"])
    y_test = np.array(test["Label"])

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)



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

    embedding_layer = Embedding(num_words+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)#



    #LSTM model
    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(SpatialDropout1D(0.2))

    model.add(LSTM(EMBEDDING_DIM,dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))#softmax

    #model.summary()#print the model

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
    #print (preds)
    acc = accuracy_score(test['Label'], preds)

    print('prediction accuracy: ', acc)

    return preds

predicted_labels=[]
#predicted_labels.append(prediction(merged_train,merged_test))

#merged_test=sep_test_headlines_1[0]
#end=prediction(merged_train,merged_test)

for p in range(25):
    merged_test=sep_test_headlines_1[p]
    single_prediction=prediction(merged_train,merged_test)
    predicted_labels.append(single_prediction)

print (len(predicted_labels),len(predicted_labels[0]))
final_label=[]

for m in range(378):
    temp=[row[m] for row in predicted_labels]
    c=Counter(temp)
    value,count=c.most_common()[0]
    final_label.append(value)


true_test_label=np.array(test["Label"])
print (true_test_label)


def acc(true,pred):
    m=0
    for i in range(len(true)):
        if(true[i]==pred[i]):
            m=m+1
        else:
            pass
    acc=m/len(true)
    return acc

final_acc=acc(true_test_label,final_label)
print (final_acc)

print ("end")
