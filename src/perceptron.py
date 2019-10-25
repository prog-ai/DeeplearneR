import os
import sys
import datetime
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.regularizers as regularizers
from src.get_data import data


# uncomment to disable GPU support
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


default_path = 'C:/Users/DimKa/Documents/'


class Perceptron(object):

    def __init__(self, path=default_path):
        self.x_train, self.y_train, self.x_test, self.y_test, self.image_shape, self.log_root, self.labels = init(path)

    def train_model(self, model, batch_size=128, n_epochs=100, optimizer=optimizers.SGD, learning_rate=1e-2):
        opt = optimizer(lr=learning_rate)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        logdir = os.path.join(self.log_root, f'{model.name}_{timestamp}')
        tensorboard_callback = callbacks.TensorBoard(logdir, profile_batch=0)
        model.fit(x=self.x_train, y=self.y_train, verbose=1, epochs=n_epochs,
                  validation_data=(self.x_test, self.y_test), batch_size=batch_size, callbacks=[tensorboard_callback])

    def multi_layer_perceptron(self):
        mlp = models.Sequential([
            layers.Flatten(input_shape=self.image_shape),
            layers.Dense(512, activation='tanh'),
            layers.Dense(10, activation='softmax')],
            name='tanh_mlp')

        return mlp

    def multi_layer_perceptron_relu(self):
        mlp_relu = models.Sequential([
            layers.Flatten(input_shape=self.image_shape),
            layers.Dense(512, activation='relu', kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(1e-3)),
            layers.Dense(10, activation='softmax')],
            name='relu_mlp_l2reg')

        return mlp_relu

    def main(self):
        if sys.argv[1] == '-r':
            mlp = self.multi_layer_perceptron_relu()
        else:
            mlp = self.multi_layer_perceptron()

        self.train_model(mlp, optimizer=optimizers.Adam, learning_rate=2e-4)


if __name__ == "__main__":
    Network = Perceptron()
    Network.main()
