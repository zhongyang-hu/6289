import keras_preprocessing.text
import tensorflow as tf
print('GPU name: ',tf.config.experimental.list_physical_devices('GPU'))
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.preprocessing import *
from keras import optimizers
from finalProject import preprocessor
from keras.initializers import Constant

data=pd.read_csv('data.csv')
from pymagnitude import *
mag = Magnitude("GoogleNews-vectors-negative300.magnitude")
data['Sentiment']=data['Sentiment'].map({'negative':0,'positive':2,'neutral':1})

xtrain,ytrain,xtest,ytest,  tokenizer, wordvectors=preprocessor(data['Sentence'].values,data['Sentiment'].values,mag,40)

model1=Sequential()


model1.add(Embedding(1+len(tokenizer.word_index), 300, embeddings_initializer=Constant(wordvectors), input_length=40, trainable=False))

model1.add(LSTM(256, activation='relu',dropout=0.2, return_sequences=True))
model1.add(LSTM(256, activation='relu',dropout=0.2, return_sequences=True))
model1.add(LSTM(256, activation='relu',dropout=0.2, return_sequences=False))

model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.25))
model1.add(Dense(3, activation='softmax'))
print(model1.summary())
model1.compile(optimizer=optimizers.Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


modelcheck = ModelCheckpoint('model1.h5', monitor='val_sparse_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
history = model1.fit(xtrain, ytrain, epochs=100, batch_size=64, validation_data=(xtest, ytest), callbacks=[modelcheck])
trainhist = pd.DataFrame(history.history)
trainhist.to_csv('model1.csv')