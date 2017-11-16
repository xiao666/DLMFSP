import numpy as np
import pandas as pd
import os
#from collections import Counter
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import np_utils
#from keras.layers import SpatialDropout1D
from keras.layers.core import Dense, Dropout, Activation, Lambda
#from keras.layers.embeddings import Embedding
from sklearn.metrics import accuracy_score
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



nb_classes=2
batch_size=32
#load X_data
#X=pd.read_csv('sum_t_prob64.csv')
#X=pd.read_csv('sum_top5.csv')
X=pd.read_csv('top5_index+1.csv')#1989,125
#X=pd.read_csv('selected_top5_index+1.csv')#1989,20
print (X.shape)

X=np.asarray(X)
#train data normalize to [0,1]

#load Y_data
data=pd.read_csv('Combined_News_DJIA.csv')
Y=data['Label']
Y=np.array(Y)
print (Y[0])
print (len(Y))


#split train and test
#X_train=X[0:1611].reshape(1611,1,64)
#X_test=X[1611:1989].reshape(378,1,64)

X_train=X[0:1611]
X_test=X[1611:1989]

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train=Y[0:1611]
y_test=Y[1611:1989]
Y_train=np_utils.to_categorical(y_train, nb_classes)
Y_test=np_utils.to_categorical(y_test, nb_classes)


#build LSTM
#modeling
print('Build model...')
model = Sequential()
model.add(Embedding(65, 128))
model.add(Dropout(0.2))
model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))#
model.add(Dense(nb_classes))
model.add(Activation('softmax'))#softmax

model.summary()#print the model

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#'binary_crossentropy''mean_squared_error' adam

print('Train...')
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=3,
          validation_data=(X_test, Y_test),callbacks=[early_stopping])
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)
acc = accuracy_score(y_test, preds)

print('prediction accuracy: ', acc)


print ("end")
