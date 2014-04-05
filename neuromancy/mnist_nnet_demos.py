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



import gzip
import theano.tensor as T

from sgd_trainer import SGDTrainer
from neural_net import LogisticRegression, MLP, LeNet5
from data_pipeline import load_data, shared_dataset


def sgd_optimize_logreg(datasets):
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels

    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    trainer = SGDTrainer(classifier, datasets)
    trainer.build(x, y)
    return trainer.train()


def sgd_optimize_mlp(datasets):
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels

    classifier = MLP(input=x, n_in=28 * 28, n_hidden=500, n_out=10)

    trainer = SGDTrainer(classifier, datasets, learning_rate=0.01, L1_reg=0.0001,
                         L2_reg=0.001, n_epochs=500, batch_size=20)
    trainer.build(x, y)
    return trainer.train()


def sgd_optimize_lenet(datasets):
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels
    batch_size = 500

    classifier = LeNet5(input=x, nkerns=[20, 50], filter_shapes=[[5, 5], [5, 5]],
                        image_shapes=[[28, 28], [12, 12]], batch_size=batch_size,
                        n_hidden=500, n_out=10)

    trainer = SGDTrainer(classifier, datasets, learning_rate=0.01, L1_reg=0.0001,
                         L2_reg=0.001, n_epochs=100, batch_size=batch_size)
    trainer.build(x, y)
    return trainer.train()


if __name__ == '__main__':

    # PREPARE DATA SETS
    #datasets = load_data('mnist.pkl.gz')
    f = gzip.open('data/kaggle_mnist.pkl.gz', 'rb')
    train_set = cPickle.load(f)
    valid_set = cPickle.load(f)
    test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]

    # TRAIN CLASSIFIER
    #logreg_classifier = sgd_optimize_logreg(datasets)
    #mlp_classifier = sgd_optimize_mlp(datasets)
    lenet_classifier = sgd_optimize_lenet(datasets)

    # SAVE TRAINED CLASSIFIER
    f = open('trained_nets/lenet5_demo.pkl')
    cPickle.dump(lenet_classifier, f, cPickle.HIGHEST_PROTOCOL)
    f.close()


    # LOAD TRAINED CLASSIFIER
    # TODO: move this code into its own module
    import numpy
    import cPickle
    import theano

    f = open('trained_nets/lenet5_demo.pkl', 'rb')
    classifier = cPickle.load(f)
    f.close()

    # LOAD TEST DATA SET
    f = open('data/test.csv', 'r')
    data = numpy.loadtxt(f, delimiter=',', skiprows=1)
    f.close()

    # COMPILE CLASSIFY FUNCTION
    # TODO: move this into a method in the NeuralNet class
    obs = T.matrix('obs')

    classify = theano.function(
        inputs=[obs],
        outputs=classifier.prediction,
        givens={classifier.input: obs.reshape((obs.shape[0], 1, 28, 28))})

    # CLASSIFY THE UNLABELED DATA
    b = 500
    n = data.shape[0]
    preds = []
    for i in xrange(n / b):
        preds.append(classify(data[i*b : (i+1)*b, :]))

    output = numpy.hstack(preds)
    output = numpy.vstack((range(1, n+1), output))  # add image id's
    output = output.transpose()
    output = output.astype(int)

    # SAVE PREDICTIONS IN A CSV FILE
    # TODO: Improve formatting: no scientific notation; include column headers
    numpy.savetxt('data/kaggle_mnist_preds_lenet.csv', output, delimiter=',')
