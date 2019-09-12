# coding: utf8
from __future__ import division

import argparse
import numpy as np
import json
import os
import pandas as pd
import pickle

from keras.models import model_from_json
from sklearn.metrics import accuracy_score


'''
In order to use this file, you need to have the model save as 'model.json' and weight save as 'model.h5' in the same folder.
'''

class Prediction():

    def __init__(self):
        '''Initialise Prediction class by loading the model and best weights saved through training.py
        '''
        self.LABEL = ['focus', 'distract']
        self.BODY_PART_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]

        # Load json and create classifier model
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
        self.model = model_from_json(loaded_model_json)

        # Load weights into the model
        self.model.load_weights("weights.best.hdf5")
        print("\nClassifier model loaded")

    def preprocess(self, json_positions, discardLowBodyPart=False):
        '''Preprocess the data to predict from json files: normalise
        Input: Json files, option to chosse the joints to select discardLowBodyPart
        Output: dataset of data to predict
        '''
        data = []

        for person in range(len(json_positions)):
            personList = []
            w = json_positions[person]['image_size']['width']
            h = json_positions[person]['image_size']['height']

            if discardLowBodyPart:
                bodyParts = [json_positions[person]['body_part'][j] for j in self.BODY_PART_INDEX]
            else:
                bodyParts = json_positions[person]['body_part']

            for bodyPart in range(len(bodyParts)):
                personList.append(bodyParts[bodyPart]['x'] / w)
                personList.append(bodyParts[bodyPart]['y'] / h)
                personList.append(bodyParts[bodyPart]['confidence'])
            data.append(np.array(personList))

        return np.array(data)

    def predict(self, X):
        '''Predict the percentage of bellongging into class 1(distract) of each person (line) in the dataset
        Intput: dataset
        Output: percentage predicted for each person (line) to belong to class 1
        '''
        self.predictions = self.model.predict(X)

        return self.predictions

    def predictClasses(self, X):
        '''Predict the label of each person (line) in the dataset
        Intput: dataset
        Output: label predicted for each person (line)
        '''
        self.predictions = self.model.predict_classes(X)

        return self.predictions


if __name__ == '__main__':
    # Start detection chain for predictions
    predictionObject = Prediction()

    X_test, y_test = pd.read_pickle('data_test.pkl')
    print('For ' + str(X_test.shape[0]) + ' persons : acc = ' + str(accuracy_score(y_test, predictionObject.predictClasses(X_test))))

    # Get argument parser
    # parser = argparse.ArgumentParser(description='Chain of focus detection using human pose detection')
    # parser.add_argument('--path', type=str, default="../pickleToPredict/", help='Path to input pickle folder')
    # args = parser.parse_args()

    # for pkl in os.listdir(args.path):
    #     data = pd.read_pickle(pkl)
    #     np.random.shuffle(data)
    #     y = data[:, -1].astype(int)
    #     x = data[:, :-2]

    #     predictions = predictionObject.predict(x)
    #     print(predictions)

    #     predictions = predictionObject.predictClasses(x)
    #     for i in range(len(predictions)):
    #         print('Person %s' %str(i+1) + ' - ' + predictionObject.LABEL[predictions[i][0]] + ' : ' + str(predictions[i]) + ' | ' + predictionObject.LABEL[y[i]])

    #     print('Acc: ', accuracy_score(y, predictions))
