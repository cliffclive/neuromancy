__author__ = 'CClive'

import theano
import numpy
import neural_layer

import theano.tensor as T

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
        self.L1_norm = self.L1_norm.sum()
        self.L2_norm = self.L2_norm.sum()

        self.prediction = T.argmax(self.output, axis=1)

        data = T.matrix('data')
        self.classify = theano.function(
            inputs=[data],
            outputs=self.prediction,
            givens={self.input: data})

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
        if y.ndim != self.prediction.ndim:
            raise TypeError('y should have the same shape as self.prediction',
                            ('y', y.type, 'y_pred', self.prediction.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.prediction, y))
        else:
            raise NotImplementedError()

    # TODO: Add a print function that displays the structure of the network.
    # TODO: Add a print function to NeuralLayer, and have NeuralNet call the
    # TODO: print function for each of its layers.


class LogisticRegression(NeuralNet):
    """
    A logistic regression is a trivial neural net that only has a single logistic layer.
    """
    def __init__(self, input, n_in, n_out):
        logreg_layer = neural_layer.LogisticLayer(input, n_in, n_out)
        super(LogisticRegression, self).__init__([logreg_layer])


class MLP(NeuralNet):
    """
    Multi-Layer Perceptron class with one hidden layer.
    TODO: generalize this to make it easy to create an arbitrary number of
    hidden layers, of arbitrary size.
    """
    def __init__(self, input, n_in, n_out, n_hiddens=[]):
        hidden_layers = []
        hidden_layers.append(neural_layer.PerceptronLayer(input, n_in, n_hiddens[0]))
        for n in range(1, len(n_hiddens)):
            hidden_layers.append(
                neural_layer.PerceptronLayer(hidden_layers[n-1].output, n_hiddens[n-1], n_hiddens[n]))

        logreg_layer = neural_layer.LogisticLayer(input=hidden_layers[-1].output,
                                                  n_in=n_hiddens[-1], n_out=n_out)
        layers = hidden_layers + [logreg_layer]
        super(MLP, self).__init__(layers)


class LeNet(NeuralNet):
    """
    Convolutional neural network with 2 conv/pool layers and 1 perceptron layer.
    """
    def __init__(self, input, nkerns, filter_shapes, image_shapes,
                 batch_size, n_hiddens, n_out, poolsize=2):
        self.filter_shapes = filter_shapes
        self.image_shapes = image_shapes
        self.batch_size = batch_size

        cp_layers = []
        input0 = input.reshape((batch_size, 1, 28, 28))
        cp_layers.append(neural_layer.LeNetConvPoolLayer(
            input=input0,
            filter_shape=(nkerns[0], 1, filter_shapes[0][0], filter_shapes[0][1]),
            image_shape=(batch_size, 1, image_shapes[0][0], image_shapes[0][1]),
            poolsize=(poolsize, poolsize)
        ))
        for n in range(1, len(nkerns)):
            cp_layers.append(
                neural_layer.LeNetConvPoolLayer(
                    input=cp_layers[n-1].output,
                    filter_shape=(nkerns[n], nkerns[n-1], filter_shapes[n][0], filter_shapes[n][1]),
                    image_shape=(batch_size, nkerns[n-1], image_shapes[n][0], image_shapes[n][1]),
                    poolsize=(poolsize, poolsize)))

        hidden_layers = []
        hidden_layers.append(neural_layer.PerceptronLayer(input=cp_layers[-1].output.flatten(2),
                                                          n_in=nkerns[1] * 4 * 4,
                                                          n_out=n_hiddens[0]))
        for n in range(1, len(n_hiddens)):
            hidden_layers.append(
                neural_layer.PerceptronLayer(hidden_layers[n-1].output, n_hiddens[n-1], n_hiddens[n]))

        logreg_layer = neural_layer.LogisticLayer(input=hidden_layers[-1].output,
                                                  n_in=n_hiddens[-1], n_out=n_out)
        super(LeNet, self).__init__(cp_layers + hidden_layers + [logreg_layer])

    def classify(self, data):
        """
        The LeNet class is only able to classify data sets that have the same dimension
        as the minibatches it is trained on. (I'm not exactly sure why; this is something
        I need to investigate.)

        This function will break up a data set into minibatches and reshape them to fit
        the right dimensions, and then collect predictions for each minibatch into a single
        array.
        """
        # TODO: Get this to work on data sets that have a number of observations
        # TODO: not equal to a multiple of the minibatch size.
        preds = []
        n = data.shape[0]
        for i in xrange(n / self.batch_size):
            batch = data[i*self.batch_size : (i+1)*self.batch_size, :]
            batch = batch.reshape((batch.shape[0], 1,
                                   self.image_shapes[0][0],
                                   self.image_shapes[0][1]))
            preds.append(super(LeNet, self).classify(batch))

        output = numpy.hstack(preds)
        return output


