__author__ = 'CClive'

import theano
import numpy

import theano.tensor as T

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


class NeuralLayer(object):
    def __init__(self, input, W=None, b=None, activation=None):
        """
        Basic layer of a neural net: weights W and bias b, along with an
        activation function, define the transformation from inputs to outputs.

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))

        # parameters of the model
        self.params = [self.W, self.b]

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1_norm = abs(self.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_norm = (self.W ** 2).sum()


class LogisticLayer(NeuralLayer):
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        W = theano.shared(value=numpy.zeros((n_in, n_out),
                                            dtype=theano.config.floatX),
                          name='W', borrow=True)
        # initialize the biases b as a vector of n_out 0s
        b = theano.shared(value=numpy.zeros((n_out,),
                                            dtype=theano.config.floatX),
                          name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        activation = T.nnet.softmax

        super(LogisticLayer, self).__init__(input, W, b, activation)


class PerceptronLayer(NeuralLayer):
    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, seed=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type seed: int
        :param seed: used to initialize a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """

        rng = numpy.random.RandomState(seed)

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        super(PerceptronLayer, self).__init__(input, W, b, activation)


class LeNetConvPoolLayer(NeuralLayer):
    """Pool Layer of a convolutional network """

    def __init__(self, input, filter_shape, image_shape, poolsize=(2, 2), seed=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type seed: int
        :param seed: used to initialize a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]

        rng = numpy.random.RandomState(seed)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX), borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)

        super(LeNetConvPoolLayer, self).__init__(input, W, b)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                               filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
