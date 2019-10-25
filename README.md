# DeeplearneR

Deep Learning models for image classification based on Convolutional Neural Networks with 92% accuracy (ResNet).

<img src="design/deep.jpg?raw=true">

## Detailed

Convolutional Neural Network model as application of Deep Learning in the context of Computer Vision and image recognition using Keras/TensorFlow. For faster and more acurate learning and prediction processes I have applied a bunch of optimizations. As additions to that, there are also packages for Softmax Regression, Multilayer Peceptron, as well as the fastest and most accurate ResNet model based on Cifra10 dataset. On my machine GPU usage helps reduce training period more than 20x, which would take arround 4 hours for 150 epochs.

<img src="design/conv.png?raw=true">


## Demo
**ResNet**

<img src="design/demo.gif?raw=true">


## Performance

<img src="design/prediction.png?raw=true">


## Package
- Convolutional Neural Networks
- ResNets
- Multilayer Perceptrons
- SoftMax Regressors

## Optimizations

- Adam or SGD optimizer
- Regularization
- Data augmentation
- Batch Normalization
- Strided Convolutions
- Global Pooling
- Learning Rate Decay


## Usage

> CNN model training is done as follows. From the outside path of the project directory use arguments -b, -s or -p for batchnormalization, strided convolutions or pooling and start training:

```shell
C:\DeeplearneR> cd ..
C:\> python -m DeeplearneR.src.cnn -s
```

> ResNet model training is done as follows. From the outside path of the project directory use argument -a to enable Data Augmentation and start training:

```shell
C:\DeeplearneR> cd ..
C:\> python -m DeeplearneR.src.resnet -a
```

## How to Contribute

1. Clone repo and create a new branch: `$ git checkout https://github.com/DumitruHanciu/DeeplearneR -b new_branch`.
2. Make changes and test
3. Submit Pull Request with comprehensive description of changes


## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**

