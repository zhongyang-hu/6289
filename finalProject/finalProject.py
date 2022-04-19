import keras_preprocessing.text
import tensorflow as tf

import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.preprocessing import *
from keras import optimizers
from keras.initializers import Constant

from pymagnitude import *

def preprocessor(data,label,mag,length):

    tokenizer=text.Tokenizer(filters='"&(),-/:;<=>[\\]_`{|}~\t\n0123456789',lower=True, split=' ')
    tokenizer.fit_on_texts(data)
    wordvectors=np.zeros((1+len(tokenizer.word_index),300))

    for w, j in tokenizer.word_index.items():
        wordvectors[j,:]=mag.query(w)

    sentence=tokenizer.texts_to_sequences(data)
    sentence=sequence.pad_sequences(sentence,length,float,'post')
    xtrain=sentence[:int(0.8*len(sentence))]
    xtest=sentence[int(0.8*len(sentence)):]
    ytrain=label[:int(0.8*len(sentence))]
    ytest=label[int(0.8*len(sentence)):]
    return xtrain,ytrain,xtest,ytest,  tokenizer, wordvectors


def balancedpreprocessor(data,mag,length):
    data['Sentiment'] = data['Sentiment'].map({'negative': 0, 'positive': 2, 'neutral': 1})
    tokenizer=text.Tokenizer(filters='"&(),-/:;<=>[\\]_`{|}~\t\n0123456789',lower=True, split=' ')
    tokenizer.fit_on_texts(data['Sentence'])
    wordvectors=np.zeros((1+len(tokenizer.word_index),300))

    for w, j in tokenizer.word_index.items():
        wordvectors[j,:]=mag.query(w)

    traindata=data[:int(0.8*len(data))].copy()
    testdata=data[int(0.8*len(data)):].copy()


    temp1=traindata[traindata['Sentiment']==1]
    n=len(temp1)
    temp2=traindata[traindata['Sentiment']==0].sample(n,replace=True)
    temp3=traindata[traindata['Sentiment']==2].sample(n,replace=True)
    trainfinal=pd.concat([temp1,temp2,temp3]).sample(int(3*n))

    xtrain=tokenizer.texts_to_sequences(trainfinal['Sentence'].values)
    xtrain=sequence.pad_sequences(xtrain,length,float,'post')
    xtest=tokenizer.texts_to_sequences(testdata['Sentence'].values)
    xtest=sequence.pad_sequences(xtest,length,float,'post')
    ytrain=trainfinal['Sentiment'].values
    ytest=testdata['Sentiment'].values
    return xtrain,ytrain,xtest,ytest,  tokenizer, wordvectors
