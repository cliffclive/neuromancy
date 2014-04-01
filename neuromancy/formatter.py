__author__ = 'CClive'

import numpy
import gzip
import cPickle
from logreg import load_data


def convert_csv(infile, outfile):
    train = numpy.loadtxt(infile, delimiter=',', skiprows=1)
    targets = train[:, 0]
    inputs = train[:, 1:]

    obs = inputs.shape[0]
    obs_t = obs * 3 / 5
    obs_v = obs * 4 / 5

    train_set = (inputs[:obs_t, :], targets[:obs_t, ])
    valid_set = (inputs[obs_t:obs_v, :], targets[obs_t:obs_v, ])
    test_set = (inputs[obs_v:, :], targets[obs_v:, ])

    datasets = (train_set, valid_set, test_set)

    f = gzip.open(outfile, 'wb')
    cPickle.dump(datasets, f, cPickle.HIGHEST_PROTOCOL)
    f.close()


if __name__ == '__main__':
    convert_csv('data/train.csv', 'data/kaggle_mnist.pkl.gz')
    datasets = load_data('kaggle_mnist.pkl.gz')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
