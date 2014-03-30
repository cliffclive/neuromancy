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


import cPickle
import gzip
import os

import numpy

import theano
import theano.tensor as T

from sgd_trainer import SGDTrainer
from neural_net import LogisticRegression, MLP, LeNet5


def load_data(dataset):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.


    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


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
                         L2_reg=0.001, n_epochs=500, batch_size=batch_size)
    trainer.build(x, y)
    return trainer.train()


if __name__ == '__main__':
    dataset='mnist.pkl.gz'
    #logreg_classifier = sgd_optimize_logreg(dataset)
    #mlp_classifier = sgd_optimize_mlp(dataset)
    lenet_classifier = sgd_optimize_lenet(dataset)


"""
kaggle_data = open('data/train.csv').readlines()[1:]
kaggle_data = [line.rstrip().split(',') for line in kaggle_data]

for i in range(len(kaggle_data)):
    kaggle_data[i] = [int(x) for x in kaggle_data[i]]

kaggle_outputs = [data[0] for data in kaggle_data]
kaggle_inputs = [data[1:] for data in kaggle_data]

kaggle_preds = mnist_logreg.y_pred(kaggle_inputs)  ## DOES NOT WORK!!

TODO:
Convert sgd_optimization_mnist into a Trainer class that accepts any classifier
and any dataset in the appropriate format. Use the "Build Model" routine in the
__init__ method, and use "Train Model"  as a method to, yes, train the model.

When the training method is finished, return the trained classifier as an object
that can be used to classify new data sets. Also write methods to save and load
classifiers, and to update classifiers that have previously been trained with
other data.

Finally, implement a way to train on streaming data, instead of data sets that are
loaded into memory all at once.
"""