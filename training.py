# -*- coding: utf-8 -*-

import tflearn
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


class Training():

    def __init__(self, data):
        # Data loading and preprocessing
        if (data.any()):
            dataset = data
        else:
            dataset = pd.read_pickle('data.pkl')
        # Remove extra data in dataset such as "source"
        # TODO: Update it with new column images size
        
        self.X = np.array(dataset)[:, :-2]
        self.y = np.array(dataset)[:, -1]

    def buildNN(self, hiddenLayerNumber=2):
        # Building deep neural network
        input_layer = tflearn.input_data(shape=[None, 54])
        dense = tflearn.fully_connected(input_layer, 54, activation='relu',
                                        regularizer='L2', weight_decay=0.001)
        for i in range(hiddenLayerNumber):
            dropout = tflearn.dropout(dense, 0.8)
            dense = tflearn.fully_connected(dropout, 54, activation='relu',
                                            regularizer='L2', weight_decay=0.001)

        dropout = tflearn.dropout(dense, 0.8)
        softmax = tflearn.fully_connected(dropout, 2, activation='softmax')

        # Regression using SGD with learning rate decay and Top-3 accuracy
        sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
        acc = tflearn.metrics.accuracy()
        #top_k = tflearn.metrics.Top_k(1)
        net = tflearn.regression(softmax, optimizer=sgd, metric=acc,
                                loss='categorical_crossentropy')
        return net

    def train(self, net):
        #split data between train and test
        X, testX, Y, testY = train_test_split(self.X, self.y, test_size=.2, random_state=42)
        Y = np.array(pd.get_dummies(Y))
        testY = np.array(pd.get_dummies(testY))

        # Training
        model = tflearn.DNN(net, tensorboard_verbose=0)
        model.fit(X, Y, n_epoch=200, validation_set=(testX, testY), show_metric=True, run_id="dense_model")

