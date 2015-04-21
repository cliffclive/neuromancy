neuromancy
==========

Neural Network implementation using Python (Numpy and theano).

This package includes wrapper classes for neural networks constructed using the theano package.

The classes can be used for quickly generating neural networks, but the code is more useful for
understanding how to build networks using theano.


See mnist_nnet_demos.py for demonstrations that show how to build and train neural networks to classify digits from
the MNIST data set.

neural_net.py provides implementations for three classes of neural networks: logistic regressions, multi-layer
perceptrons, and the LeNet classifier.

sgd_trainer.py provides a class used for training networks by stochastic gradient descent.