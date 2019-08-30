# -*- coding: utf-8 -*-

""" Deep Neural Network for MNIST dataset classification task.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tflearn
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Data loading and preprocessing
dataset = pd.read_pickle('data.pkl')
X = np.array(dataset)[:,:-2]
y = np.array(dataset)[:,-1] 

#split data between train and test
X, testX, Y, testY = train_test_split(X, y, test_size=.2, random_state=42)


Y = np.array(pd.get_dummies(Y))
testY = np.array(pd.get_dummies(testY))



# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 54])
dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
softmax = tflearn.fully_connected(dropout2, 2, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(1)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=20, validation_set=(testX, testY), show_metric=True, run_id="dense_model")

