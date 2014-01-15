'''
Created on Jan 10, 2014

@author: Clive
'''


import math
import numpy as np


def sigmoid(array):
    return 1.0 / (1.0 + np.exp(-array))


def sigmoid_gradient(array):
    sig = sigmoid(array)
    return sig * (1 - sig)


def add_bias(array):
    bias = np.ones((array.shape[0], 1))
    return np.hstack((bias, array))


def binarize_categories(targets, categories):
    """
    Converts a binary array of integers into a binary matrix with as many columns
    as the maximum integer in the array, with a 1 in the kth column of each row
    where the integer value is k.
    """
    output = np.zeros((len(targets), len(categories)))
    for i in range(len(targets)):
        output[i, categories[targets[i]]] = 1
    return output


class FeedForwardBackPropNetwork:
    def __init__(self, n_features, n_classes, hidden_layer_sizes, learning_rate, regularization=0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_layers = len(hidden_layer_sizes) + 2
        self.hidden_layer_sizes = hidden_layer_sizes
        self.signals = [0] * self.n_layers
        self.activations = [0] * self.n_layers
        self.weights = self.randomize_weights()
        self.deltas = [0] * self.n_layers
        self.gradients = [wt * 0 for wt in self.weights]
        self.learning_rate = learning_rate
        self.regularization = regularization

    def randomize_weights(self):
        sizes = [self.n_features] + self.hidden_layer_sizes + [self.n_classes]
        weights = []
        for l in range(len(sizes)-1):
            shape = (sizes[l] + 1, sizes[l+1])
            epsilon = math.sqrt(6) / math.sqrt(shape[0] + shape[1])
            wt = np.random.uniform(-epsilon, epsilon, shape)
            weights.append(wt)
        return weights

    def feed_forward(self, inputs):
        self.signals = [inputs]
        self.activations = [inputs]
        for wt in self.weights:
            signal = add_bias(self.activations[-1]).dot(wt)
            activation = sigmoid(signal)
            self.signals.append(signal)
            self.activations.append(activation)

    def count_observations(self):
        return self.activations[0].shape[0]

    def probabilities(self):
        return self.activations[-1]

    def calculate_deltas(self, targets):
        self.deltas[-1] = self.probabilities() - targets
        for i in reversed(range(self.n_layers - 1)):
            self.deltas[i] = ((self.deltas[i+1].dot(self.weights[i].transpose())) *
                              add_bias(sigmoid_gradient(self.signals[i])))[:, 1:]

    def calculate_gradients(self):
        for i in range(len(self.gradients)):
            self.gradients[i] = (add_bias(self.activations[i]).transpose()).dot(self.deltas[i+1])
            self.gradients[i] /= self.count_observations()

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.gradients[i]

    def calculate_cost(self, targets):
        n = self.count_observations()
        outputs = self.probabilities()
        base_cost = (-targets * np.log(outputs) -
                     (1-targets) * np.log(1-outputs)).sum() / n
        reg_cost = self.regularization / (2 * n) * (
            sum([(np.power(wt, 2)).sum() for wt in self.weights]))
        return base_cost + reg_cost

    def predict(self, inputs):
        self.feed_forward(inputs)
        return np.argmax(self.probabilities(), 1)

    def learn_weights(self, data, targets, tolerance=0.0001, max_iter=10000, show_progress=False):
        self.feed_forward(data)
        cost = self.calculate_cost(targets)
        print "Initial Cost: ", cost
        d_cost = 100
        i = 0
        while (abs(d_cost) > tolerance) and (i < max_iter):
            self.calculate_deltas(targets)
            self.calculate_gradients()
            self.update_weights()
            self.feed_forward(data)
            new_cost = self.calculate_cost(targets)
            d_cost = cost - new_cost
            cost = new_cost
            i += 1
        if show_progress:
            print "Final Cost: ", cost
            print "Iterations: ", i
            print "Max Iterations Reached: ", (i == max_iter)


if __name__ == "__main__":
    # Move these into Unit Tests
    
    #print sigmoid_gradient(0)
    #print sigmoid_gradient(100)
    #print sigmoid_gradient(-100)

    X1 = np.array([[1, 2, 1, 2, 3],
                   [2, 4, 2, 3, 4],
                   [3, 6, 3, 4, 5],
                   [4, 8, 4, 5, 6],
                   [5, 0, 5, 6, 7],
                   [6, 3, 6, 7, 8],
                   [7, 6, 7, 8, 9],
                   [8, 8, 8, 9, 0]])
    Y1 = np.array([[0, 1],
                   [1, 0],
                   [0, 1],
                   [1, 0],
                   [1, 0],
                   [0, 1],
                   [1, 0],
                   [0, 1]])

    X2 = np.array([[3, 6, 3, 4, 5],
                   [4, 8, 4, 5, 6],
                   [5, 0, 5, 6, 7],
                   [6, 3, 6, 7, 8]])
    Y2 = np.array([[0, 1],
                   [1, 0],
                   [1, 0],
                   [0, 1]])

    f = X1.shape[1]
    k = Y1.shape[1]
    a = 0.2
    r = 1
    hidden = [4, 3]
    ffbp_net = FeedForwardBackPropNetwork(f, k, hidden, a, r)

    """
    ffbp_net.feed_forward(X1)
    print "Cost(1): ", ffbp_net.calculate_cost(Y1)

    ffbp_net.calculate_deltas(Y1)
    ffbp_net.calculate_gradients()
    ffbp_net.update_weights()

    ffbp_net.feed_forward(X1)
    print "Cost(2): ", ffbp_net.calculate_cost(Y1)
    """

    ffbp_net.learn_weights(X1, Y1, 0.00001, 1000)

    """
    # Make sure it works with a different sized training set
    ffbp_net.feed_forward(X2)
    ffbp_net.calculate_deltas(Y2)
    print ffbp_net.deltas[-1]
    print "Cost(2): ", ffbp_net.calculate_cost(Y2)
    """
    
    # End unit tests
    
    # Begin Kaggle tests

    print "\n\nTesting on MNIST data..."
    import pandas as pd

    print "Reading data..."
    batch_size = 1000
    data_dir = "C:/Users/Clive/Dropbox/code/pynets/pynets/Data/"
    train_file = pd.read_csv(data_dir + "train.csv", chunksize=batch_size)
    train = train_file.get_chunk()
    digits = dict(zip(range(10), range(10)))
    Y = binarize_categories(train['label'], digits)
    X = np.array(train.ix[:, 1:])

    print "Building network..."
    f = X.shape[1]
    k = Y.shape[1]
    a = 0.2
    r = 1
    hidden = [100, 50, 50, 100]
    digits_net = FeedForwardBackPropNetwork(f, k, hidden, a, r)

    print "Training network..."
    digits_net.learn_weights(X, Y, 0.00001, 1000)
    i=1
    while True:
        try:
            train = train_file.get_chunk()
        except StopIteration:
            break
        Y = binarize_categories(train['label'], digits)
        X = np.array(train.ix[:, 1:])
        digits_net.learn_weights(X, Y, 0.00001, 1000, show_progress=False)
        
    test_file = pd.read_csv(data_dir + "test.csv")
    X = np.array(test_file)
    pred = digits_net.predict(X)
    pred = pd.DataFrame(pred)
    pred.to_csv(data_dir + "predictions.csv")

