__author__ = 'CClive'

import theano
import numpy

import theano.tensor as T

from neural_layer import NeuralLayer

class NeuralNet(object):
    """
    Generic neural network class
    Acts as a container that manages a list of neural layers.
    Also adds cost, error, and prediction functions to the network.
    """
    def __init__(self, layers):
        self.layers = layers
        self.input = self.layers[0].input
        self.output = self.layers[-1].output

        self.params = []
        self.L1_norm = []
        self.L2_norm = []
        for layer in self.layers:
            self.params += layer.params
            self.L1_norm += layer.L1_norm
            self.L2_norm += layer.L2_norm

        self.prediction = T.argmax(self.output, axis=1)

    def cost(self, y, L1_reg=0, L2_reg=0):
        """
        Uses the negative log likelihood for the cost function. This is appropriate
        when the last layer of the network is a logistic regression layer, which is
        a popular choice. Override this method if another cost function is desired.

        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        prediction_cost = -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])
        regularization_cost = L1_reg * self.L1_norm + L2_reg * self.L2_norm
        return prediction_cost + regularization_cost

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        # what is 'target.type'?
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
