import os
import sys
import datetime
import numpy as np
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.regularizers as regularizers
from .get_data import data
from .plot import plot_multiple
from .comp_implem import my_predict_cnn

# uncomment to disable GPU support
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

default_path = 'C:/Users/DimKa/Documents/'

class CNN(object):

    def __init__(self, path=default_path):
        self.x_train, self.y_train, self.x_test, self.y_test, self.image_shape, self.log_root, self.labels = data(path)

    def train_model(self, model, batch_size=128, n_epochs=100, optimizer=optimizers.SGD, learning_rate=1e-2):
        opt = optimizer(lr=learning_rate)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        logdir = os.path.join(self.log_root, f'{model.name}_{timestamp}')
        tensorboard_callback = callbacks.TensorBoard(logdir, profile_batch=0)
        model.fit(x=self.x_train, y=self.y_train, verbose=1, epochs=n_epochs,
                  validation_data=(self.x_test, self.y_test), batch_size=batch_size, callbacks=[tensorboard_callback])

    def compare_cnn(self, cnn):
        w1, b1 = cnn.layers[0].get_weights()
        w2, b2 = cnn.layers[2].get_weights()
        w3, b3 = cnn.layers[5].get_weights()

        i_test = 1
        inp = self.x_test[i_test]
        my_prob = my_predict_cnn(inp, w1, b1, w2, b2, w3, b3)
        keras_prob = cnn.predict(inp[np.newaxis])[0]
        if np.mean((my_prob - keras_prob) ** 2) > 1e-10:
            print('Something isn\'t right! Keras gives different results than my_predict_cnn!')
        else:
            print('Congratulations, you got correct results!')

        i_maxpred = np.argmax(my_prob)
        plot_multiple([self.im_test[i_test]], [f'Pred: {self.labels[i_maxpred]}, {my_prob[i_maxpred]:.1%}'], imheight=2)

    def convolutional_neural_network(self):
        cnn = models.Sequential([
            layers.Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same',
                          input_shape=self.image_shape, kernel_regularizer=regularizers.l2(1e-3)),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same',
                          kernel_regularizer=regularizers.l2(1e-3)),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Flatten(),
            layers.Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(1e-3))],
            name='cnn')

        return cnn

    def convolutional_neural_network_batchnorm(self):
        cnn_batchnorm = models.Sequential([
            layers.Conv2D(64, (3, 3), use_bias=False, padding='same', input_shape=self.image_shape),
            layers.BatchNormalization(scale=False),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2), (2, 2)),
            layers.Conv2D(64, (3, 3), use_bias=False, padding='same'),
            layers.BatchNormalization(scale=False),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2), (2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(1e-3))],
            name='cnn_batchnorm')

        return cnn_batchnorm

    def convolutional_neural_network_strides(self):
        cnn_strides = models.Sequential([
            layers.Conv2D(64, 3, strides=2, use_bias=False, padding='same', input_shape=self.image_shape),
            layers.BatchNormalization(scale=False),
            layers.Activation('relu'),
            layers.Conv2D(64, 3, strides=2, use_bias=False, padding='same'),
            layers.BatchNormalization(scale=False),
            layers.Activation('relu'),
            layers.Flatten(),
            layers.Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(1e-3))],
            name='cnn_strides')

        return cnn_strides

    @staticmethod
    def convolutional_neural_network_pooling(self):
        cnn_global_pool = models.Sequential([
            layers.Conv2D(64, 3, 2, padding='same', use_bias=False),
            layers.BatchNormalization(scale=False),
            layers.Activation('relu'),
            layers.Conv2D(64, 3, 2, padding='same', use_bias=False),
            layers.BatchNormalization(scale=False),
            layers.Activation('relu'),
            layers.Conv2D(64, 3, padding='same', use_bias=False),
            layers.BatchNormalization(scale=False),
            layers.Activation('relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(1e-3))],
            name='cnn_global_pool')

        return cnn_global_pool

    def main(self):
        if 2 <= len(sys.argv) < 3:
            if sys.argv[1] == '-b':
                cnn = self.convolutional_neural_network_batchnorm()
            elif sys.argv[1] == '-s':
                cnn = self.convolutional_neural_network_strides()
            elif sys.argv[1] == '-p':
                cnn = self.convolutional_neural_network_pooling()
            else:
                cnn = self.convolutional_neural_network()
        else:
            cnn = self.convolutional_neural_network()

        self.train_model(cnn, optimizer=optimizers.Adam, learning_rate=1e-3)

        self.compare_cnn(cnn)


if __name__ == "__main__":
    Network = CNN()
    Network.main()
