__author__ = 'CClive'
"""
These examples demonstrate how to build, train, save, load, and make predictions
from neural networks built from the NeuralNet class in neural_net.py.

In these samples, we use data from the MNIST data set, using data sets that can be
downloaded from deeplearning.net, or from the digit recognizer competition on
kaggle.com.

Building and training the neural nets follows a simple pattern:
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = NeuralNet(input=x, n_in=28 * 28, n_out=10, ...)

    trainer = SGDTrainer(classifier, datasets, ...)
    trainer.build(x, y)
    return trainer.train()

"""



import gzip
import numpy
import theano
import cPickle
import theano.tensor as T

from sgd_trainer import SGDTrainer
from neural_net import LogisticRegression, MLP, LeNet
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

    classifier = MLP(input=x, n_in=28 * 28, n_out=10, n_hiddens=[1500])

    trainer = SGDTrainer(classifier, datasets, learning_rate=0.01, L1_reg=0.0001,
                         L2_reg=0.001, n_epochs=500, batch_size=20)
    trainer.build(x, y)
    return trainer.train()


def sgd_optimize_lenet(datasets):
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels

    classifier = LeNet(input=x, nkerns=[20, 50], filter_shapes=[[5, 5], [5, 5]],
                        image_shapes=[[28, 28], [12, 12]], batch_size=500,
                        n_hidden=[500], n_out=10)

    trainer = SGDTrainer(classifier, datasets, learning_rate=0.01,
                         L1_reg=0.0001, L2_reg=0.001, n_epochs=100,
                         batch_size=classifier.batch_size)
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
    logreg_classifier = sgd_optimize_logreg(datasets)
    #mlp_classifier = sgd_optimize_mlp(datasets)
    #lenet_classifier = sgd_optimize_lenet(datasets)

    '''
    # SAVE TRAINED CLASSIFIER
    f = open('trained_nets/lenet5_demo.pkl')
    cPickle.dump(lenet_classifier, f, cPickle.HIGHEST_PROTOCOL)
    f.close()

    # LOAD TRAINED CLASSIFIER
    f = open('trained_nets/lenet5_demo.pkl', 'rb')
    classifier = cPickle.load(f)
    f.close()
    '''

    # LOAD TEST DATA SET
    f = open('data/test.csv', 'r')
    data = numpy.loadtxt(f, delimiter=',', skiprows=1)
    f.close()
    n = data.shape[0]

    preds = logreg_classifier.classify(data)

    # FORMAT THE PREDICTIONS IN PREPARATION FOR SAVING TO FILE
    # TODO: put this into its own function, or a class
    output = numpy.hstack(preds)
    output = numpy.vstack((range(1, n+1), output))  # add image id's
    output = output.transpose()
    output = output.astype(int)

    # SAVE PREDICTIONS IN A CSV FILE
    # TODO: Improve formatting: no scientific notation; include column headers
    numpy.savetxt('data/kaggle_mnist_preds_logreg.csv', output, delimiter=',')
    #numpy.savetxt('data/kaggle_mnist_preds_mlp.csv', output, delimiter=',')
    #numpy.savetxt('data/kaggle_mnist_preds_lenet.csv', output, delimiter=',')
