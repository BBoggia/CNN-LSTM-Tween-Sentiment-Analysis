import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D
from keras.layers import LSTM, Bidirectional
from keras.layers import Flatten, Dropout
from keras.models  import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention
import visualkeras

########################## Parameters ##########################
embedding_units = 20000
max_sequence_length = 150
conv1d_filter_cound = 32
conv1d_kernal_size = 4
conv1d_activation = 'relu'
lstm_units = 150
lstm_dropout = 0.2
lstm_recurrent_dropout = 0.2
attention_activation='sigmoid'
dense_neuron_count = 1
dense_activation='sigmoid'

train_shuffle = False
train_epochs = 80
train_batch_size = 32
######################################################################

df = pd.read_csv('./Tweets.csv')

def process_data(_df):
    print(_df.shape)
    df = _df[_df['airline_sentiment'] != 'neutral']
    print(df.shape)
    df = df[["tweet_id", "text", "airline_sentiment"]]

    df['text'] = df['text'].str.replace('@\S+', '')
    df['airline_sentiment'] = df['airline_sentiment'].map({'negative': 0, 'positive': 1})

    return df

df_processed = process_data(df)

def split_data(df):
    X = df['text'].values
    y = df['airline_sentiment'].values

    X_train, y_train = X[:int(len(X) * 0.8)], y[:int(len(y) * 0.8)]
    X_test, y_test = X[int(len(X) * 0.8):], y[int(len(y) * 0.8):]

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = split_data(df_processed)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print("X Train: ", len(X_train))
print("X Test: ", len(X_test))

max_len = 150

X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

print("Shape of X Train: ", X_train.shape)
print("Shape of X Test: ", X_test.shape)

model = Sequential()
model.add(Embedding(20000, max_len, input_length=max_len))
model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(Bidirectional(LSTM(150, dropout=0.2, return_sequences=True)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, shuffle=False, validation_data=(X_test, y_test), epochs=3, batch_size=196)
#model.save_weights(checkpoint_path)

#visualkeras.layered_view(model, to_file='output.png', legend=True, scale_xy=0.5, scale_z=1, max_z=300).show()  # font is optional!

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

tweet = "I am so happy and joyful after that flight"
tweet = tokenizer.texts_to_sequences([tweet])
tweet = sequence.pad_sequences(tweet, maxlen=150)
print(model.predict(tweet))

tweet = "I hate this movie"
tweet = tokenizer.texts_to_sequences([tweet])
tweet = sequence.pad_sequences(tweet, maxlen=150)
print(model.predict(tweet))

tweet = "That flight was terrible"
tweet = tokenizer.texts_to_sequences([tweet])
tweet = sequence.pad_sequences(tweet, maxlen=150)
print(model.predict(tweet))

tweet = "I had a great time"
tweet = tokenizer.texts_to_sequences([tweet])
tweet = sequence.pad_sequences(tweet, maxlen=150)
print(model.predict(tweet))

tweet = "Get out of the door"
tweet = tokenizer.texts_to_sequences([tweet])
tweet = sequence.pad_sequences(tweet, maxlen=150)
print(model.predict(tweet))

