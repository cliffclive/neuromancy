__author__ = 'CClive'

import theano
import numpy
import time
import sys
import os

import theano.tensor as T


class SGDTrainer(object):
    def __init__(self, classifier, datasets,
                 learning_rate=0.13, L1_reg=0, L2_reg=0,
                 n_epochs=1000, batch_size=600):
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        # compute number of minibatches for training, validation and testing
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0] / batch_size

        self.classifier = classifier
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def build(self, x, y):
        print '... building the model'

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
        self.test_model = theano.function(
            inputs=[index],
            outputs=self.classifier.errors(y),
            givens={
                x: self.test_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                y: self.test_set_y[index * self.batch_size:(index + 1) * self.batch_size]})

        self.validate_model = theano.function(
            inputs=[index],
            outputs=self.classifier.errors(y),
            givens={
                x: self.valid_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                y: self.valid_set_y[index * self.batch_size:(index + 1) * self.batch_size]})

        # the cost we minimize during training is the negative log likelihood
        # of the model plus the regularization terms (L1 and L2);
        # cost is expressed here symbolically
        cost = self.classifier.cost(y, L1_reg, L2_reg)

        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=cost, wrt=self.classifier.W)
        g_b = T.grad(cost=cost, wrt=self.classifier.b)

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(self.classifier.W, self.classifier.W - self.learning_rate * g_W),
                   (self.classifier.b, self.classifier.b - self.learning_rate * g_b)]

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size]})

    def train(self):
        print '... training'
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(self.n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatches before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_params = None
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        while (epoch < self.n_epochs) and (not done_looping):
            epoch += 1
            for minibatch_index in xrange(self.n_train_batches):
                minibatch_avg_cost = self.train_model(minibatch_index)

                # iteration number
                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in xrange(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                                improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [self.test_model(i) for i in xrange(self.n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

        return self.classifier
