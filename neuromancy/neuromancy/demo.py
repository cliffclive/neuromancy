'''
Created on Jan 16, 2014

@author: Clive
'''

import pandas as pd
from neuromancy import FeedForwardBackPropNetwork

if __name__ == "__main__":
    
    # The following is a routine for implementing mini-batch learning.
    # I plan to make a general class for building, training, and evaluating
    # different machine learning algorithms, including the neural networks
    # classes in the neuromancy module.
    
    print "Reading data..."
    batch_size = 1000
    data_dir = "C:/Data/kaggle/digits/"
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
