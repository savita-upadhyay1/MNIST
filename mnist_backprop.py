"""
Author: Savita Upadhyay
Date: 26th April 2022
"""

import random
import datetime
import numpy as np
import plotly.graph_objects as go
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report


def init_weight_vector(neuron_count, num_features):
    """
    This function initializes input and hidden weights.
    :param neuron_count:
    :param num_features:
    :return:
    """
    input_weights = []
    hidden_weights = []
    # input_bias = []
    # hidden_bias = []

    # input weights per neuron for all samples :
    # matrix of dimension :[neuron][784:variables in input data]

    for wt in range(0, neuron_count, 1):
        input_weights.append([random.uniform(-0.05, 0.05)
                              for i in range(num_features + 1)])
        # input_bias.append([random.uniform(-0.05, 0.05) for i in range(1)])

    # hidden weights per output for all neurons :
    # matrix of dimension :[10:output][neuron count]
    for wt in range(0, 10, 1):
        hidden_weights.append([random.uniform(-0.05, 0.05)
                               for i in range(neuron_count + 1)])
        # hidden_bias.append([random.uniform(-0.05, 0.05) for i in range(1)])

    return np.array(input_weights), np.array(hidden_weights)


def add_column_1(inputs: np.array):
    """
    This is utility function to add one to array.
    :param inputs:
    :return:
    """
    # Function2 : add column(all 1) for bias
    ones = np.ones((np.array(inputs).shape[0], 1))
    inputs = np.concatenate((ones, inputs), axis=1)

    return inputs


def sigmoid(z: np.array):
    """
    Function to calculate sigmoid.
    :param z:
    :return:
    """
    return 1.0 / (1.0 + np.exp(-z))


def one_hot_encode(digit: int) -> []:
    """
    Utility function to encode a passed digit and return one hot encoded array
    :param digit:
    :return:
    """
    y_label = np.zeros(10)
    y_label[digit] = 1
    return y_label


def forward_propagate(input_weights, hidden_weights, train):
    """
    Fucntion used for forward propogation using input and hidden weights
    :param input_weights:
    :param hidden_weights:
    :param train:
    :return:
    """
    # pass each sample as an array
    sample = np.array(train)

    # activation for input layer
    transposed_sample = np.transpose(sample)  # .reshape(784 , )
    Z1 = np.matmul(input_weights, transposed_sample)
    # Z1 = Z1.reshape(20,)
    hidden_neuron = sigmoid(Z1)

    # print(hidden_weights.shape, hidden_neuron.shape)
    # print(hidden_neuron)
    hidden_neuron = np.append([1.0], hidden_neuron)
    # print(hidden_neuron)

    # print(hidden_weights.shape, hidden_neuron.shape)
    # activation for hidden layer
    Z2 = np.matmul(hidden_weights, hidden_neuron)
    Z2 = Z2.reshape(10, )
    # print(Z2)

    output = sigmoid(Z2)
    return Z1, hidden_neuron.reshape(hidden_neuron.shape[0], ), Z2, output


"""
========================================================================================================================
Functions for Backpropagation. 
========================================================================================================================
"""


def delta_hidden_to_out(output, target):
    """
    Calcuate delta for hidden layer to output units used in backprop,
    :param output:
    :param target:
    :return:
    """
    output = output
    delta1 = output * ((1 - output) * (target - output))

    return delta1


def delta_input_to_hidden(delta1, hidden_weights, hidden_neuron):
    """
    Function used to calcuate delta for input to hidden layer used in backpropr
    :param delta1:
    :param hidden_weights:
    :param hidden_neuron:
    :return:
    """
    delta2 = []
    for j in range(len(hidden_neuron)):
        wkj = np.sum(hidden_weights[:, j] * delta1)
        delta = hidden_neuron[j] * (1 - hidden_neuron[j]) * wkj
        delta2.append(delta)

    return np.array(delta2)


def update_hidden_weights(delta_k: [], hidden_forward: [],
                          hidden_weights: [], delta_wt_t1: [],
                          eta: float = 0.2, momentum: float = 0.9):
    """
    Function used to update weights from hidden layer to output weights.
    :param delta_k:
    :param hidden_forward:
    :param hidden_weights:
    :param delta_wt_t1:
    :param eta:
    :param momentum:
    :return:
    """
    # print('In update_hidden_weights')
    # Update weights
    delta_wt_t = []

    # 10x1, 1x20
    delta_ho_tranposed = delta_k.reshape(1, delta_k.shape[0]).transpose()
    hidden_forward_reshaped = hidden_forward.reshape(1, hidden_forward.shape[0])
    delta_j_mul_hj = np.matmul(delta_ho_tranposed, hidden_forward_reshaped)

    delta_wt_kj = eta * delta_j_mul_hj
    delta_wt_kj += momentum * delta_wt_t1
    # print(hidden_weights[0])
    # Update hidden wts
    hidden_weights = hidden_weights + delta_wt_kj
    # print(delta_wt_kj[0])
    # print(hidden_weights[0])
    # Save delta wt kj
    delta_wt_t1 = delta_wt_kj
    return delta_wt_t1, hidden_weights


def update_input_weights(delta_j: [], input_weights: [], sample: [],
                         delta_ji_prev: [], eta: float = 0.2,
                         momentum: float = 0.9):
    """
    Function used to update weights from input to hidden layer used in backprop.
    :param delta_j:
    :param input_weights:
    :param sample:
    :param delta_ji_prev:
    :param eta:
    :param momentum:
    :return:
    """
    # print('In input_weights')
    # Update weights

    delta_j = delta_j.reshape(1, delta_j.shape[0]).transpose()
    x_i = sample
    # print(delta_j.shape, x_i.shape)
    delta_j_mul_x_i = np.matmul(delta_j, x_i)

    delta_ji = eta * delta_j_mul_x_i
    momentum_term = momentum * delta_ji_prev
    input_weights = input_weights + delta_ji + momentum_term

    delta_wt_i1 = delta_ji
    return delta_wt_i1, input_weights


"""
========================================================================================================================
Functions for Prediction. 
========================================================================================================================
"""


def prediction(output, target):
    """
    This function compared the the prediction output with target label and return 1 if the prediction is correct.
    :param output:
    :param target:
    :return:
    """
    prediction_idx = np.argmax(output)
    actual_idx = np.argmax(target)
    count = 0
    if prediction_idx == actual_idx:
        count += 1

    return count, prediction_idx


def test_predict(y_test_subset, x_test_vec, input_weights, hidden_weights):
    """
    This methods predicts on the test dataset and returns accuracy and arrays for confusion matrix.
    :param y_test_subset:
    :param x_test_vec:
    :param input_weights:
    :param hidden_weights:
    :return:
    """
    test_labelchecks = []
    predictions = []
    gt = []
    for i in range(len(x_test_vec)):
        x_test = add_column_1([x_test_vec[i]])
        Z1, hidden_neuron, Z2, output = forward_propagate(input_weights,
                                                          hidden_weights,
                                                          x_test)

        # test_output=outputs.index(max(outputs))
        target = one_hot_encode(y_test_subset[i])
        test_labelcheck, pred_digit = prediction(output, target)

        test_labelchecks.append(test_labelcheck)
        gt.append(y_test_subset[i])
        predictions.append(pred_digit)

    test_labelchecks = np.sum(test_labelchecks)
    test_accuracy = (100 * test_labelchecks) / 10000
    return test_accuracy, gt, predictions


"""
========================================================================================================================
Functions for Training for one epoch. 
========================================================================================================================
"""


def train(x_train_vec, y_train_subset,
          input_weights, hidden_weights, eta, momentum):
    """
    This is function which iterates over all samples in training dataset, runs forward prop,
    calculates errors, then back propogates errors, and also keeps track of incorrect predictions.
    :param x_train_vec:
    :param y_train_subset:
    :param input_weights:
    :param hidden_weights:
    :param eta:
    :param momentum:
    :return:
    """
    label_check = []
    delta_wt_t1 = np.zeros(hidden_weights.shape)
    delta_wt_inpts = np.zeros(input_weights.shape)

    for sample_idx in range(0, len(x_train_vec)):
        sample = [x_train_vec[sample_idx]]
        target = y_train_subset[sample_idx]

        sample = add_column_1(sample)
        target = one_hot_encode(target)

        # Forward propogate
        Z1, hidden_neuron, Z2, output = forward_propagate(input_weights,
                                                          hidden_weights,
                                                          sample)

        # Update hidden --> Out Wts
        delta_k = delta_hidden_to_out(output, target)
        delta_wt_t1, hidden_weights = update_hidden_weights(
            delta_ho=delta_k,
            hidden_forward=hidden_neuron,
            hidden_weights=hidden_weights,
            delta_wt_t1=delta_wt_t1,
            eta=eta,
            momentum=momentum)

        # Update input --> hidden wts
        delta_j = delta_input_to_hidden(delta_k, hidden_weights,
                                        hidden_neuron)
        delta_wt_input, input_weights = update_input_weights(delta_j[1:],
                                                             input_weights,
                                                             sample, delta_wt_inpts,
                                                             eta=eta, momentum=momentum)

        count, prediction_idx = prediction(output, target)
        label_check.append(count)
        # print(np.sum(input_weights), np.sum(hidden_weights))

    return np.array(label_check), input_weights, hidden_weights


"""
========================================================================================================================
Functions for Training for all epochs 
========================================================================================================================
"""


def train_for_epochs(input_weights: [], hidden_weights: [],
                     x_train: np.array, y_train: np.array,
                     x_test: np.array, y_test: np.array,
                     epochs: int = 50, num_samples=60000,
                     eta=0.2, momentum=0.9):
    """
    This is basically used to call the train function per epoch and calculaete test epoch once training finishes
    for one epoch.
    """
    training_accuracy = []
    test_accuracy = []
    input_weights_t = input_weights
    hidden_weights_t = hidden_weights

    # Run for multiple epochs
    print('Started at: ', datetime.datetime.now())

    for epoch in range(0, epochs, 1):
        # print(np.sum(input_weights_t), np.sum(hidden_weights_t))

        accuracy_check, input_weights_t, hidden_weights_t = \
            train(x_train[0:num_samples],
                  y_train[0:num_samples],
                  input_weights_t,
                  hidden_weights_t,
                  eta, momentum)

        # print(np.sum(input_weights_t), np.sum(hidden_weights_t))
        # print(np.sum(accuracy_check), num_samples)
        training_acrcy = (100 * np.sum(accuracy_check)) / len(accuracy_check)
        training_accuracy.append(training_acrcy)

        test_acrcy, gt, predictions = test_predict(y_test, x_test,
                                                   input_weights_t,
                                                   hidden_weights_t)
        test_accuracy.append(test_acrcy)

    print('Ended at: ', datetime.datetime.now())
    return (training_accuracy, test_accuracy), (gt, predictions)


"""
========================================================================================================================
Functions for Loading data 
========================================================================================================================
"""


def load_data():
    """
    Function used to load MNIST data for test and train datasets.
    :return:
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train1 = x_train
    x_test1 = x_test
    x_train = x_train / 255
    x_test = x_test / 255

    x_train.shape

    train_indices = np.argwhere(y_train < 10)
    test_indices = np.argwhere(y_test < 10)

    # print(y_train.dtype)
    # print(np.max(y_train))
    y_train_subset = np.int8(y_train[np.where(y_train < 10)])
    # y_train_subset[y_train_subset == 0] = -1

    y_test_subset = np.int8(y_test[np.where(y_test < 10)])
    # y_test_subset[y_test_subset == 0] = -1

    x_train_vec = np.take(x_train, train_indices, axis=0)
    x_train_vec = x_train_vec.reshape(x_train_vec.shape[0], 28 * 28)

    x_test_vec = np.take(x_test, test_indices, axis=0)
    x_test_vec = x_test_vec.reshape(x_test_vec.shape[0], 28 * 28)

    # print(x_train_vec.shape)
    # print(y_train_subset.shape)
    # print(x_test_vec.shape)
    # print(y_test_subset.shape)

    return (x_train_vec, y_train_subset), (x_test_vec, y_test_subset)


def equisampling(classize):
    """
    Function used to load MNIST data by selecting equal number of samples per digit class and shuffle the inputs.
    :param classize:
    :return:
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_vec1 = []
    y_train_subset1 = []

    x_train = x_train / 255
    x_test = x_test / 255

    test_indices = np.argwhere(y_test < 10)
    y_test_subset = np.int8(y_test[np.where(y_test < 10)])

    x_test_vec = np.take(x_test, test_indices, axis=0)
    x_test_vec = x_test_vec.reshape(x_test_vec.shape[0], 28 * 28)

    for i in range(10):
        train_indices = np.argwhere(y_train == i)
        y_train_subset = np.int8(y_train[np.where(y_train == i)])[0:classize]
        y_train_subset1.extend(y_train_subset)

        x_train_vec = np.take(x_train, train_indices, axis=0)[0:classize]
        x_train_vec = x_train_vec.reshape(x_train_vec.shape[0], 28 * 28)
        x_train_vec1.extend(x_train_vec)

    idx = np.random.permutation(len(y_train_subset1))
    x_train_vec1 = np.array(x_train_vec1)[idx]
    y_train_subset1 = np.array(y_train_subset1)[idx]

    return x_train_vec1, y_train_subset1.reshape(10 * classize, ), x_test_vec, y_test_subset


"""
========================================================================================================================
Function for Plotting Errors 
========================================================================================================================
"""


def plot_errors(train, test):
    """
    Generate Plots for training and test accuracy.
    :param train:
    :param test:
    :return:
    """
    fig = go.Figure()
    trace1 = go.Scatter(
        x=np.arange(0, len(train)),
        y=train,
        name='Training'
    )
    trace2 = go.Scatter(
        x=np.arange(0, len(test)),
        y=test,
        name='Test'
    )

    fig.add_trace(trace1)
    fig.add_trace(trace2)

    fig.update_layout(yaxis=dict(title='% of incorrect predictions'),
                      xaxis=dict(title='Epoch #'),
                      title='Training/Test Error per epoch',
                      )
    fig.show()

    return fig


"""
========================================================================================================================
Functions for Experiment 1 and code block to call experiment 1 and plot
========================================================================================================================
"""


def experiment1(x_train_vec, y_train_subset, x_test_vec, y_test_subset,
                neuron_counts=[20],
                num_features=784, epochs=50,
                num_samples=60000):
    print('Running experiment 1')

    accuracy_map = {}
    for neuron_count in neuron_counts:
        input_weights, hidden_weights = init_weight_vector(neuron_count,
                                                           num_features)

        (training_accuracy, test_accuracy), (gt, predictions) = \
            train_for_epochs(input_weights=input_weights,
                             hidden_weights=hidden_weights,
                             x_train=x_train_vec, y_train=y_train_subset,
                             x_test=x_test_vec, y_test=y_test_subset,
                             epochs=epochs, num_samples=num_samples,
                             eta=0.2, momentum=0.9)

        accuracy_map[neuron_count] = [(training_accuracy, test_accuracy),
                                      (gt, predictions)]

    return accuracy_map


# Run experiment1 and create plots and report accuracy metrics
(x_train_vec, y_train_subset), (x_test_vec, y_test_subset) = load_data()
experiment_accuracy_map = experiment1(x_train_vec, y_train_subset,
                                      x_test_vec, y_test_subset,
                                      neuron_counts=[20, 50, 100],
                                      epochs=50, num_samples=60000)

for key in experiment_accuracy_map.keys():
    print('Neuron count: ', key)
    (training_accuracy, test_accuracy) = experiment_accuracy_map[key][0]
    (gt, predictions) = experiment_accuracy_map[key][1]
    print('Confusion Matrix: ')
    print(confusion_matrix(gt, predictions))
    print(classification_report(gt, predictions))

    training_error = [100 - x for x in training_accuracy]
    test_error = [100 - x for x in test_accuracy]
    fig = plot_errors(training_error, test_error)


