#transfromer model
import pandas as pd

import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.preprocessing import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant


data=pd.read_csv('data.csv')
from pymagnitude import Magnitude
mag = Magnitude("GoogleNews-vectors-negative300.magnitude")



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



data=pd.read_csv('data.csv')
from pymagnitude import *
mag = Magnitude("GoogleNews-vectors-negative300.magnitude")
data['Sentiment']=data['Sentiment'].map({'negative':0,'positive':2,'neutral':1})

xtrain,ytrain,xtest,ytest,  tokenizer, wordvectors=preprocessor(data['Sentence'].values,data['Sentiment'].values,mag,40)

embed_dim = 300
num_heads = 8
ff_dim = 128

maxlen=40

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim,wordvectors):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=300, embeddings_initializer=Constant(wordvectors),trainable=False)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions





class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim,rate=0.2):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

model=Sequential()
model.add(Input(shape=(maxlen,)))
model.add(TokenAndPositionEmbedding(40, 1+len(tokenizer.word_index), 300,wordvectors))
model.add(TransformerBlock(embed_dim, num_heads, ff_dim))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(3, activation="softmax"))

print(model.summary())
t=time.time()
model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
modelcheck = ModelCheckpoint('model4.h5', monitor='val_sparse_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
history = model.fit(xtrain, ytrain, epochs=100, batch_size=64, validation_data=(xtest, ytest), callbacks=[modelcheck])
trainhist = pd.DataFrame(history.history)
trainhist.to_csv('model4.csv')
print('training time:',time.time()-t)