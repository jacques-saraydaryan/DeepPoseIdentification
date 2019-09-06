import argparse
import numpy as np
import json
import os
import pickle

from keras.models import model_from_json

'''
In order to use this file, you need to have the model save as 'model.json' and weight save as 'model.h5' in the same folder. 
'''

class Prediction():

    def __init__(self):
        self.LABEL = ['focus', 'distract']
        self.BODY_PART_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]

        # Load json and create classifier model
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
        self.model = model_from_json(loaded_model_json)

        # Load weights into the model
        self.model.load_weights("model.h5")
        print("\nClassifier model loaded")

    def preprocess(self, json_positions, discardLowBodyPart=False):
        data = []

        with open('mean_std.pkl', 'rb') as f:
            mean, std = pickle.load(f)
            f.close()

        for person in range(len(json_positions)):
            personList = []
            w = json_positions[person]['imageSize']['width']
            h = json_positions[person]['imageSize']['height']
            
            if discardLowBodyPart:
                bodyParts = [json_positions[person]['body_part'][j] for j in self.BODY_PART_INDEX]
            else:
                bodyParts = json_positions[person]['body_part']
            
            for bodyPart in range(len(bodyParts)):
                personList.append(bodyParts[bodyPart]['x'] / w)
                personList.append(bodyParts[bodyPart]['y'] / h)
                personList.append(bodyParts[bodyPart]['confidence'])
            data.append((np.array(personList) - mean) / std)

        return np.array(data)


    def predict(self, X):
        self.predictions = self.model.predict(X)

        return self.predictions
    
    def predictClasses(self, X):
        self.predictions = self.model.predict_classes(X)

        return self.predictions


if __name__ == '__main__':
    # Get argument parser
    parser = argparse.ArgumentParser(description='Chain of focus detection using human pose detection')
    parser.add_argument('pathToJson', type=str, help='Path to input json folder')
    args = parser.parse_args()

    # # Open json file
    with open(args.pathToJson, 'r') as f:
        json_positions = json.load(f)
        f.close()

    ## Start detection chain for predictions

    predictionObject = Prediction()

    data = predictionObject.preprocess(json_positions)

    predictions = predictionObject.predict(data)
    print(predictions)
    
    predictions = predictionObject.predictClasses(data)
    for i in range(len(predictions)):
        print('Person %s' %str(i+1) + ' - ' + predictionObject.LABEL[predictions[i][0]] + ' : ' + str(predictions[i]))
