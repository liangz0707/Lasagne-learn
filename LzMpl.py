# coding:utf-8
import lasagne
import theano
import theano.tensor as T
import numpy as np
__author__ = 'liangz14'


def build_mlp(input_var):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var, name=None)

    l_in_drop = lasagne.layers.DropoutLayer(incoming=l_in, p=0.2)

    non_lin = lasagne.nonlinearities.rectify

    l_hid1 = lasagne.layers.DenseLayer(incoming=l_in_drop
                                       , num_units=800
                                       , nonlinearity=non_lin
                                       , b=lasagne.init.Constant(0.0)
                                       , W=lasagne.init.GlorotUniform())

    l_hid1_drop = lasagne.layers.DropoutLayer(incoming=l_hid1, p=.5)

    l_hid2 = lasagne.layers.DenseLayer(incoming=l_hid1_drop
                                       , num_units=800
                                       , nonlinearity=non_lin
                                       , b=lasagne.init.Constant(0.)
                                       , W=lasagne.init.GlorotUniform())

    l_hid2_drop = lasagne.layers.DropoutLayer(incoming=l_hid2, p=.5)

    l_out = lasagne.layers.DenseLayer(incoming=l_hid2_drop
                                      , num_units=10
                                      , nonlinearity=lasagne.nonlinearities.softmax
                                      , b=lasagne.init.Constant(.0)
                                      , W=lasagne.init.GlorotUniform())

    return l_out


def load_dataset():

    import gzip

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(inputs, targets, batchsize, shuffle):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in xrange(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


def main(num_epochs=3):

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print("Constructing network...")
    network = build_mlp(input_var=input_var)

    prediction = lasagne.objectives.get_output(network)
    # ? categorical_crossentropy=True
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    param = lasagne.layers.get_all_params(network)
    update = lasagne.updates.nesterov_momentum(loss_or_grads=loss
                                               , params=param
                                               , learning_rate=0.01
                                               , momentum=0.9)

    test_pred = lasagne.layers.get_output(network
                                          , deterministic=True)  # ? deterministic=True
    test_loss = lasagne.objectives.categorical_crossentropy(test_pred
                                                            , targets=target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_pred, axis=1), target_var)
                      , dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=update)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("Starting Training...")
    import time
    for epoch in xrange(num_epochs):
        train_err = 0
        tarin_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            tarin_batches += 1
            print "训练已完成%-.2f%%" % (tarin_batches*500.0/len(X_train)*100)
        print train_err*1.0 / tarin_batches
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

if __name__== '__main__':
    main();