#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Input, Flatten, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report 
from gensim.models import word2vec
import logging

import nltk, string


def cnn_model(FILTER_SIZES,               # filter sizes as a list
              MAX_NB_WORDS, \
              # total number of words
              MAX_DOC_LEN, \
              # max words in a doc
              EMBEDDING_DIM=200, \
              # word vector dimension
              NUM_FILTERS=64, \
              # number of filters for all size
              DROP_OUT=0.5, \
              # dropout rate
              NUM_OUTPUT_UNITS=1, \
              # number of output units
              NUM_DENSE_UNITS=100,\
              # number of units in dense layer
              PRETRAINED_WORD_VECTOR=None,\
              # Whether to use pretrained word vectors
              LAM=0.0):            
              # regularization coefficient
    
                main_input = Input(shape=(MAX_DOC_LEN,),                            dtype='int32', name='main_input')

                if PRETRAINED_WORD_VECTOR is not None:
                    embed_1 = Embedding(input_dim=MAX_NB_WORDS+1,                             output_dim=EMBEDDING_DIM,                             input_length=MAX_DOC_LEN,                             # use pretrained word vectors
                            weights=[PRETRAINED_WORD_VECTOR],\
                            # word vectors can be further tuned
                            # set it to False if use static word vectors
                            trainable=True,\
                            name='embedding')(main_input)
                else:
                    embed_1 = Embedding(input_dim=MAX_NB_WORDS+1,                             output_dim=EMBEDDING_DIM,                             input_length=MAX_DOC_LEN,                             name='embedding')(main_input)
        # add convolution-pooling-flat block
                conv_blocks = []
                for f in FILTER_SIZES:
                    conv = Conv1D(filters=NUM_FILTERS, kernel_size=f,                                   activation='relu', name='conv_'+str(f))(embed_1)
                    conv = MaxPooling1D(MAX_DOC_LEN-f+1, name='max_'+str(f))(conv)
                    conv = Flatten(name='flat_'+str(f))(conv)
                    conv_blocks.append(conv)

                if len(conv_blocks)>1:
                    z=Concatenate(name='concate')(conv_blocks)
                else:
                    z=conv_blocks[0]

                drop=Dropout(rate=DROP_OUT, name='dropout')(z)

                dense = Dense(NUM_DENSE_UNITS, activation='relu',                                kernel_regularizer=l2(LAM),name='dense')(drop)
                preds = Dense(NUM_OUTPUT_UNITS, activation='sigmoid', name='output')(dense)
                model = Model(inputs=main_input, outputs=preds)

                model.compile(loss="binary_crossentropy",                           optimizer="adam", metrics=["accuracy"]) 

                return model

def detect_duplicate(datafile):
    data = pd.read_csv(datafile)
    
    #defining cnn_model


    # training wordvector for model 
    sentences_1=[ [token.strip(string.punctuation).strip()              for token in nltk.word_tokenize(doc)                  if token not in string.punctuation and                  len(token.strip(string.punctuation).strip())>=2]             for doc in data['q1']]
    sentences_2=[ [token.strip(string.punctuation).strip()              for token in nltk.word_tokenize(doc)                  if token not in string.punctuation and                  len(token.strip(string.punctuation).strip())>=2]             for doc in data['q2']]
            
    sentences = sentences_1 + sentences_2

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',                         level=logging.INFO)
    EMBEDDING_DIM=200
    # min_count: words with total frequency lower than this are ignored
    # size: the dimension of word vector
    # window: is the maximum distance 
    #         between the current and predicted word 
    #         within a sentence (i.e. the length of ngrams)
    # workers: # of parallel threads in training
    # for other parameters, check https://radimrehurek.com/gensim/models/word2vec.html
    wv_model = word2vec.Word2Vec(sentences,                                  min_count=5,                                  size=EMBEDDING_DIM,                                  window=5, workers=4 )
    
    
    EMBEDDING_DIM=200
    MAX_NB_WORDS=1000

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(data[["q1", "q2"]])
    # tokenizer.word_index provides the mapping 
    # between a word and word index for all words
    NUM_WORDS = min(MAX_NB_WORDS, len(tokenizer.word_index))
#     print(len(tokenizer.word_index))

    # "+1" is for padding symbol
    embedding_matrix = np.zeros((NUM_WORDS+1, EMBEDDING_DIM))
    # print(NUM_WORDS)
    for word, i in tokenizer.word_index.items():
        # if word_index is above the max number of words, ignore it
        if i >= NUM_WORDS:
            continue
        if word in wv_model.wv:
            embedding_matrix[i]=wv_model.wv[word]
            
            
    EMBEDDING_DIM=1000
    FILTER_SIZES=[2,3,4]

    # set the number of output units
    # as the number of classes
    # output_units_num=len(mlb.classes_)

    #Number of filters for each size
    num_filters=64
    MAX_DOC_LEN=500
    MAX_NB_WORDS=10000
    # EMBEDDING_DIM=200
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(data["q1"])

    # set the dense units
    dense_units_num= num_filters*len(FILTER_SIZES)

    BTACH_SIZE = 32
    NUM_EPOCHES = 100

    sequences_1 = tokenizer.    texts_to_sequences(data["q1"])
    # print(sequences_1)

    sequences_2 = tokenizer.    texts_to_sequences(data["q2"])

    sequences =  sequences_1 + sequences_2

    output_units_num=1



    # pad all sequences into the same length 
    # if a sentence is longer than maxlen, pad it in the right
    # if a sentence is shorter than maxlen, truncate it in the right
    padded_sequences = pad_sequences(sequences,                                      maxlen=MAX_DOC_LEN,                                      padding='post',                                      truncating='post')


    X_train, X_test, Y_train, Y_test = train_test_split(                    padded_sequences[0:500], data['is_duplicate'],                     test_size=0.2, random_state=0,                     shuffle=True)



    model=cnn_model(FILTER_SIZES, MAX_NB_WORDS,                     MAX_DOC_LEN,                     NUM_FILTERS=num_filters,                    NUM_OUTPUT_UNITS=output_units_num,                     NUM_DENSE_UNITS=dense_units_num,                     PRETRAINED_WORD_VECTOR = None)

    # create the model with embedding matrix
    left_cnn=cnn_model(FILTER_SIZES, MAX_NB_WORDS,                     MAX_DOC_LEN,                     NUM_FILTERS=num_filters,                    NUM_OUTPUT_UNITS=output_units_num,                     NUM_DENSE_UNITS=dense_units_num,                    PRETRAINED_WORD_VECTOR= None)

    right_cnn=cnn_model(FILTER_SIZES, MAX_NB_WORDS,                     MAX_DOC_LEN,                     NUM_FILTERS=num_filters,                    NUM_OUTPUT_UNITS=output_units_num,                     NUM_DENSE_UNITS=dense_units_num,                    PRETRAINED_WORD_VECTOR= None)

    model.summary()
    
    
    BEST_MODEL_FILEPATH="best_model"
    earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
    checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_loss',                                  verbose=2, save_best_only=True, mode='min')

    training=model.fit(X_train, Y_train,               batch_size=BTACH_SIZE, epochs=NUM_EPOCHES,               callbacks=[earlyStopping, checkpoint],              validation_data=[X_test, Y_test], verbose=2)
    
    
    pred=model.predict(X_test)
    # print(pred)
    Y_pred=np.copy(pred)
    Y_pred=np.where(Y_pred>0.5,1,0)

    # Y_pred[0:10]
    # Y[500:510]

    print(classification_report(Y_test,                     Y_pred, target_names=['0', '1']))
    
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
if __name__ == "__main__":
    detect_duplicate("quora_duplicate_question_500.csv")

