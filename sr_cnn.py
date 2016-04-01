# coding:utf-8
import lasagne
import theano
import theano.tensor as T
import time
import numpy as np
__author__ = 'liangz14'


def build_sr_cnn(input_var, input_shape):
    """
        由于输入时一组patch所以filter_size不能太大。
    :param input_var:
    :param input_shape:
    :return:
    """
    l_in = lasagne.layers.InputLayer(shape=input_shape
                                     , input_var=input_var)

    # l_in_drop = lasagne.layers.DropoutLayer(incoming=l_in, p=.2)

    l_conv1 = lasagne.layers.Conv2DLayer(incoming=l_in
                                         , num_filters=80
                                         , filter_size=(3, 3)
                                         , nonlinearity=lasagne.nonlinearities.linear
                                         #, nonlinearity=lasagne.nonlinearities.rectify
                                         , W=lasagne.init.GlorotUniform())

    l_maxpool1 = lasagne.layers.MaxPool2DLayer(l_conv1
                                               , pool_size=(2, 2))

    l_dropout1 = lasagne.layers.DropoutLayer(l_maxpool1
                                             , p=.15)

    l_full1 = lasagne.layers.DenseLayer(l_dropout1
                                        , num_units=256
                                        , nonlinearity=lasagne.nonlinearities.linear
                                        )

    l_dropout2 = lasagne.layers.DropoutLayer(l_full1
                                             , p=.15)

    network = lasagne.layers.DenseLayer(l_dropout2,
            num_units=9,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

import cPickle
def load_dataset(tag="cnn_1"):
    train_file = open('./tmp_file/_%s_training_data.pickle' % tag, 'rb')
    training_data = cPickle.load(train_file)
    train_file.close()
    target_lib, feature_lib = training_data
    target_lib = [target_patch[3:6, 3:6].reshape((9,)) for target_patch in target_lib]
    training_data = (target_lib, feature_lib)
    return training_data


def save_param(network):
    params = lasagne.layers.get_all_param_values(network)
    f = open('sr_cnn.save', 'wb')
    cPickle.dump(params, f)
    f.close()
    return


def load_param(network):
    f = open('sr_cnn.save', 'rb')
    params = cPickle.load(f)
    f.close()
    lasagne.layers.set_all_param_values(network, params)
    return params


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def training(network, feature_var, target_var, feature, target, num_epochs=500):
    feature_train = np.asarray(feature[:-1000:], dtype=theano.config.floatX)
    target_train = np.asarray(target[:-1000:], dtype=theano.config.floatX)
    feature_val = np.asarray(feature[-1000::], dtype=theano.config.floatX)
    target_val = np.asarray(target[-1000::], dtype=theano.config.floatX)
    feature_test = np.asarray(feature[::1000], dtype=theano.config.floatX)
    target_test = np.asarray(target[::1000], dtype=theano.config.floatX)

    print "========================Training...=========================="
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.sgd(
            loss, params, learning_rate=0.01)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()

    train_fn = theano.function([feature_var, target_var], loss, updates=updates)
    val_fn = theano.function([feature_var, target_var], test_loss)

    print("Starting training...")

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(feature_train, target_train, 500, shuffle=True):
            inputs, targets = batch
            train_err = train_fn(inputs, targets)
            train_batches += 1
            print "训练已完成%-.2f%% ：误差为:%-.2f" % (train_batches*500.0/len(feature_train)*100,train_err)

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(feature_val, target_val, 500, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    save_param(network)
    print "Training Finished!"
    return


def loading(network):
    print "========================Loading Model...=========================="
    load_param(network)
    print "Loading Model Finished!"
    return


def main():
    target_lib, feature_lib = load_dataset()

    input_var = T.dtensor4("feature_patchs")
    target_var = T.dmatrix("target_patchs3*3")

    input_shape = (None, 8, 9, 9)
    network = build_sr_cnn(input_var=input_var, input_shape=input_shape)

    print lasagne.layers.get_output_shape(network)
    print target_lib[0].shape
    start = time.time()
    training(network, input_var, target_var, feature_lib, target_lib)
    end = time.time()
    print "训练耗时%d秒" % (end - start)


    save_param(network)

if __name__ == "__main__":
    main()











