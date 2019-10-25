import os
import sys
import datetime
import numpy as np
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.optimizers as optimizers
from .get_data import data
from .plot import plot_multiple

# uncomment to disable GPU support
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

default_path = 'C:/Users/DimKa/Documents/'


def learning_rate_schedule(epoch):
    """Learning rate is scheduled to be reduced after 80 and 120 epochs.
    This function is automatically every epoch as part of callbacks
    during training.
    """

    if epoch < 80:
        return 1e-3
    if epoch < 120:
        return 1e-4
    return 1e-5


class softmax(object):

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

    def print_tmpl(self, softmax_regression):
        w, b = softmax_regression.layers[1].get_weights()
        templates = w.reshape(32, 32, 3, 10).transpose(3, 0, 1, 2)
        mini = np.min(templates, axis=(1, 2, 3), keepdims=True)
        maxi = np.max(templates, axis=(1, 2, 3), keepdims=True)
        rescaled_templates = (templates - mini) / (maxi - mini)
        plot_multiple(rescaled_templates, self.labels, max_columns=5, imwidth=1, imheight=1)

    def softmax_regression(self, optimizer, print_templates=False):
        softmax_regression = models.Sequential([
            layers.Flatten(input_shape=self.image_shape),
            layers.Dense(10, activation='softmax')], name='linear')

        if optimizer == optimizers.SGD:
            learning_rate = 1e-2
        else:
            learning_rate = 2e-4

        self.train_model(softmax_regression, optimizer=optimizer, learning_rate=learning_rate)

        if print_templates:
            self.print_tmpl(softmax_regression)

    def main(self):
        if sys.argv[1] == '-a':
            optimizer = optimizers.Adam
        else:
            optimizer = optimizers.SGD

        self.softmax_regression(optimizer, print_templates=True)


if __name__ == "__main__":
    Network = softmax()
    Network.main()
