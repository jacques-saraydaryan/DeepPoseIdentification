import argparse
import tflearn
import numpy as np

from processsing import Processing
from training import Training

class Prediction():

    def __init__(self, data=None):
        # Construct the Neural Network classifier and start the learning phase
        training = Training(data)
        net = training.buildNN()
        self.model = tflearn.DNN(net, tensorboard_verbose=0)

    def predict(self, data):
        self.X = np.array(data)[:, :-2]
        self.model.load('DNN.tflearn')
        predictions = self.model.predict(data)

        return predictions

if __name__ == '__main__':
    # Get argument parser
    parser = argparse.ArgumentParser(description='Chain of focus detection using human pose detection')
    parser.add_argument('path', type=str, default='../openPoseDatasetPredict/', help='Path to input json dataset')
    args = parser.parse_args()

    ## Start detection chain for predictions

    # Concat all the positions data into a single array and save it as pickle file
    process = Processing()
    data = process.createInputMatrix(args.path)
    data = process.standardise(data)

    # Prediction
    prediction = Prediction()
    print(prediction.predict(data))
