import argparse
import tflearn
import numpy as np

from processsing import Processing
from training import Training

class Prediction():

    def __init__(self):
        # Construct the Neural Network classifier and start the learning phase
        training = Training()
        net = training.buildNN()
        self.model = tflearn.DNN(net, tensorboard_verbose=0)
        self.LABEL = ['focus', 'distract']

    def predict(self, data):
        self.model.load('./DNN.tflearn', weights_only=True)
        X = np.array(data)[:, :-2]

        predictions = self.model.predict(X)

        return predictions

    def getMostProbableLabel(self, prediction):
        result = np.where(prediction == np.amax(prediction))
        return self.LABEL[result[0][0]]

if __name__ == '__main__':
    # Get argument parser
    parser = argparse.ArgumentParser(description='Chain of focus detection using human pose detection')
    parser.add_argument('--path', type=str, default='../openPoseDatasetPredict/', help='Path to input json dataset')
    args = parser.parse_args()

    ## Start detection chain for predictions

    # Concat all the positions data into a single array and save it as pickle file
    process = Processing()
    data = process.createInputMatrix(args.path)
    data = process.standardise(data)

    # Prediction
    prediction = Prediction()
    predictions = prediction.predict(data)
    for index, pred in enumerate(predictions):
        print('Personne n°' + str(index) + ' is ' + prediction.getMostProbableLabel(pred))
