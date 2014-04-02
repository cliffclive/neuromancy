__author__ = 'CClive'

import numpy
import theano
import cPickle
import gzip
import os

import theano.tensor as T


def convert_labeled_csv_data(infile, outfile):
    # Only do it this way when the data set is to big to pickle all at once,
    # (but still small enough to pickle in three batches).
    # This isn't the smartest approach; we still have the case where we can
    # pickle all sets in a single tuple, and the case where each set is still
    # to big to pickle. These require a single pickle dump, and a streaming
    # pickle dump, respectively. Or we can just focus on three pickles and a
    # streaming pickle. But the mnist data set from the deeplearning site is
    # bigger than the kaggle one. So there must be a way to pickle the kaggle
    # set in a single dump.
    print '... loading data into numpy array'
    train = numpy.loadtxt(infile, delimiter=',', skiprows=1)
    targets = train[:, 0]
    inputs = train[:, 1:]

    print '... splitting data into train, validation, and test sets'
    obs = inputs.shape[0]
    obs_t = obs * 3 / 5
    obs_v = obs * 4 / 5

    train_set = (inputs[:obs_t, :], targets[:obs_t, ])
    valid_set = (inputs[obs_t:obs_v, :], targets[obs_t:obs_v, ])
    test_set = (inputs[obs_v:, :], targets[obs_v:, ])

    #datasets = (train_set, valid_set, test_set)

    print '... pickling data sets'
    f = gzip.open(outfile, 'wb')
    #cPickle.dump(datasets, f, cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(train_set, f, cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(valid_set, f, cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(test_set, f, cPickle.HIGHEST_PROTOCOL)
    f.close()


def convert_unlabeled_csv_data(infile, outfile):
    print '... loading data into numpy array'
    test = numpy.loadtxt(infile, delimiter=',', skiprows=1)

    print '... pickling test data set'
    f = gzip.open(outfile, 'wb')
    cPickle.dump(test, f, cPickle.HIGHEST_PROTOCOL)
    f.close()


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


def load_data(dataset):
    """
    Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """

    #############
    # LOAD DATA #
    #############

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

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def download_dataset(filename='mnist.pkl.gz', folder='data',
                     url='http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'):
    dataset = os.path.join(folder, filename)
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], folder, dataset)
        if os.path.isfile(new_path) or data_file == filename:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == filename:
        import urllib
        origin = url
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)


if __name__ == '__main__':
    #convert_csv('data/train.csv', 'data/kaggle_mnist.pkl.gz')
    datasets = load_data('data/kaggle_mnist.pkl.gz')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
