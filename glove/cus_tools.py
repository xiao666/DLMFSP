#from glove import Glove, Corpus
import pandas as pd
import numpy as np

#print (sys.version)#3.6

def output():
    print ("cus_tools linked...")
    pass


###load corpus words
def load_embedding(glove_name):
    print('Indexing word vectors.')

    embeddings_index = {}       #build an word embedding dictionary
    f = open(glove_name,'r',encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


###load sample paragraph
def load_text(text_name):
    print('Processing text dataset')
    train_headlines,test_headlines=[],[]

    data = pd.read_csv('Combined_News_DJIA.csv')
    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']
    for row in range(0,len(train.index)):
        train_headlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
    
    for row in range(0,len(test.index)):
        testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
        
    return 0

def prediction(merged_train,merged_test):
    maxlen=MAX_SEQUENCE_LENGTH = 20
    max_features=MAX_NB_WORDS = 10000
    EMBEDDING_DIM = 100
    #VALIDATION_SPLIT = 0.2
    batch_size = 32
    nb_classes = 2


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
    print (preds)
    acc = accuracy_score(test['Label'], preds)

    print('prediction accuracy: ', acc)

    return predicted_Y
