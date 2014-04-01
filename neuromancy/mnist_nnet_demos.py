"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}

The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)

This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.

References:
    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2
"""
__docformat__ = 'restructedtext en'



import numpy
import theano
import theano.tensor as T

from sgd_trainer import SGDTrainer
from neural_net import LogisticRegression, MLP, LeNet5
from data_pipeline import load_data


def sgd_optimize_logreg(dataset='mnist.pkl.gz'):
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels

    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    datasets = load_data(dataset)
    trainer = SGDTrainer(classifier, datasets)
    trainer.build(x, y)
    return trainer.train()


def sgd_optimize_mlp(dataset='mnist.pkl.gz'):
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels

    classifier = MLP(input=x, n_in=28 * 28, n_hidden=500, n_out=10)

    datasets = load_data(dataset)
    trainer = SGDTrainer(classifier, datasets, learning_rate=0.01, L1_reg=0.0001,
                         L2_reg=0.001, n_epochs=500, batch_size=20)
    trainer.build(x, y)
    return trainer.train()


def sgd_optimize_lenet(dataset='mnist.pkl.gz'):
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels
    batch_size = 500

    classifier = LeNet5(input=x, nkerns=[20, 50], filter_shapes=[[5, 5], [5, 5]],
                        image_shapes=[[28, 28], [12, 12]], batch_size=batch_size,
                        n_hidden=500, n_out=10)

    datasets = load_data(dataset)
    trainer = SGDTrainer(classifier, datasets, learning_rate=0.01, L1_reg=0.0001,
                         L2_reg=0.001, n_epochs=100, batch_size=batch_size)
    trainer.build(x, y)
    return trainer.train()


if __name__ == '__main__':
    dataset='mnist.pkl.gz'
    #logreg_classifier = sgd_optimize_logreg(dataset)
    #mlp_classifier = sgd_optimize_mlp(dataset)
    lenet_classifier = sgd_optimize_lenet(dataset)

