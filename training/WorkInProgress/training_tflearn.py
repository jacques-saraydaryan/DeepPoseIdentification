import argparse
import tflearn
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from processsing import Processing


class Training():

    def __init__(self, data=None):
        # Data loading and preprocessing
        if data is not None or data.any():
            dataset = data
        else:
            dataset = pd.read_pickle('data.pkl')
        # Remove extra data in dataset such as "source"

        self.X = np.array(dataset)[:, :-2]
        self.y = np.array(dataset)[:, -1]
        self.featureNb = self.X.shape[1]

    def buildNN(self, hiddenLayerNumber=3):
        # Building deep neural network
        input_layer = tflearn.input_data(shape=[None, self.featureNb])
        dense = tflearn.fully_connected(input_layer, self.featureNb, activation='tanh',
                                        regularizer='L2', weight_decay=0.001)
        for i in range(hiddenLayerNumber):
            dropout = tflearn.dropout(dense, 0.8)
            dense = tflearn.fully_connected(dropout, self.featureNb, activation='tanh', bias=True,
                                            regularizer='L2', weight_decay=0.001)

        dropout = tflearn.dropout(dense, 0.8)
        softmax = tflearn.fully_connected(dropout, 2, activation='softmax', bias=True)
        logistic = tflearn.fully_connected(dropout, 2, activation='sigmoid', bias=True)

        # Regression using SGD with learning rate decay and Top-3 accuracy
        sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=1000)
        acc = tflearn.metrics.accuracy()
        #top_k = tflearn.metrics.Top_k(1)
        net = tflearn.regression(logistic, optimizer=sgd, metric=acc,
                                loss='categorical_crossentropy')
        return net

    def train(self, net, epochNb):
        #split data between train and test
        X, testX, Y, testY = train_test_split(self.X, self.y, test_size=.2, random_state=42)
        Y = np.array(pd.get_dummies(Y))
        testY = np.array(pd.get_dummies(testY))

        # Training
        model = tflearn.DNN(net, tensorboard_dir='./tensorboard/', tensorboard_verbose=0)
        model.fit(X, Y, n_epoch=epochNb, batch_size=32, validation_set=(testX, testY), validation_batch_size=32, show_metric=True, run_id="dense_model", shuffle=True)

        model.save('./DNN.tflearn')

if __name__ == '__main__':

    # Get argument parser
    parser = argparse.ArgumentParser(description='Chain of focus detection using human pose detection')
    parser.add_argument('--path', type=str, default='../openPoseDataset/', help='Path to input json dataset')
    parser.add_argument('--epochNb', type=int, default=500, help='Number of epoch wanted to train the NN')
    args = parser.parse_args()

    ## Start detection chain for training

    # Discard some of the lower body part to be more precise
    discardLowBodyPart = True

    # Concat all the positions data into a single array and save it as pickle file
    process = Processing()
    data = process.createInputMatrix(args.path, discardLowBodyPart)
    pickleFileName = args.path.split('/')[-2] + '.pkl'
    data = process.standardise(data, pickleFileName)

    # Construct the Neural Network classifier and start the learning phase
    training = Training(data)
    net = training.buildNN(4)

    # Train the Neural Network
    training.train(net, args.epochNb)
