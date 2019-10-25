import numpy as np

def conv3x3_same(x, weights, biases):
    """Convolutional layer with filter size 3x3 and 'same' padding.
    `x` is a NumPy array of shape [height, width, n_features_in]
    `weights` has shape [3, 3, n_features_in, n_features_out]
    `biases` has shape [n_features_out]
    Return the output of the 3x3 conv (without activation)
    """
    x = np.pad(x, [(1, 1), (1, 1), (0, 0)], mode='constant')
    result = np.empty((x.shape[0], x.shape[1], weights.shape[3]))
    for i in range(1, x.shape[0] - 1):
        for j in range(1, x.shape[1] - 1):
            roi = x[i - 1:i + 2, j - 1:j + 2]
            for c_out in range(weights.shape[3]):
                result[i, j, c_out] = np.sum(roi * weights[..., c_out])

    result = result[1:-1, 1:-1]
    result += biases
    return result


def maxpool2x2(x):
    """Max pooling with pool size 2x2 and stride 2.
    `x` is a numpy array of shape [height, width, n_features]
    """
    result = np.empty((x.shape[0] // 2, x.shape[1] // 2, x.shape[2]))
    for i in range(0, result.shape[0]):
        for j in range(0, result.shape[1]):
            roi = x[i * 2:i * 2 + 2, j * 2:j * 2 + 2]
            result[i, j] = np.max(roi, axis=(0, 1))
    return result


def dense(x, weights, biases):
    return x.T @ weights + biases


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    maxi = np.max(x)
    exponentiated = np.exp(x - maxi)
    return exponentiated / np.sum(exponentiated)


def my_predict_cnn(x, w1, b1, w2, b2, w3, b3):
    x = conv3x3_same(x, w1, b1)
    x = relu(x)
    x = maxpool2x2(x)
    x = conv3x3_same(x, w2, b2)
    x = relu(x)
    x = maxpool2x2(x)
    x = x.reshape(-1)
    x = dense(x, w3, b3)
    x = softmax(x)
    return x

