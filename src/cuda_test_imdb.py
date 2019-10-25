from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

imdb = keras.datasets.imdb

feat = 20000
maxlen = 80

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=feat)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen = maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen = maxlen)

model = keras.models.Sequential()
model.add(keras.layers.Embedding(feat, 128))
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(512, return_sequences=True))
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(512, return_sequences=True))
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(512, return_sequences=True))
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(512, return_sequences=True))
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(512))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=2, validation_data=(x_test, y_test))
