import json
import pickle
import glob
import argparse
import numpy as np
from sklearn import preprocessing

# Run on Python 3 only
# the recursive option of the glob module is not available on Python 2

class Processing():

    def __init__(self):
        # Constants
        self.LABEL = ['focus', 'distract']
        self.SOURCE = ['google', 'perso']
        self.BODY_PART_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]

    def createInputMatrix(self, path, discardLowBodyPart=False):
        files = [f for f in glob.glob(path + "**/*.*", recursive=True)]
        data = []
        print('Process the following json files :\n')
        for i, json_path in enumerate(files):
            print(json_path)
            # Extract source and label from json path
            source, label = json_path.split('/')[-3: -1]

            # Open json file
            with open(json_path, 'r') as f:
                json_positions = json.load(f)
                f.close()

            # Create the input matrix required for the neural network
            for i in range(len(json_positions)):
                personList = []
                w = json_positions[i]['imageSize']['width']
                h = json_positions[i]['imageSize']['height']

                if discardLowBodyPart:
                    bodyParts = [json_positions[i]['body_part'][j] for j in self.BODY_PART_INDEX]
                else:
                    bodyParts = json_positions[i]['body_part']

                for j in range(len(bodyParts)):
                    personList.append(bodyParts[j]['x'] / w)
                    personList.append(bodyParts[j]['y'] / h)
                    personList.append(bodyParts[j]['confidence'])

                personList.append(self.SOURCE.index(source))
                personList.append(self.LABEL.index(label))
                data.append(personList)

        return data

    def standardise(self, dataset, pickleFileName):
        # Separate in two datasets: with label 0 and label 1
        data = np.array(dataset)

        print('Mean : ' + str(data[:, :-2].mean(0)))
        print('STD : ' + str(data[:, :-2].std(0)))

        with open('mean_std.pkl', 'wb') as f:
            pickle.dump([data[:,:-2].mean(0), data[:, :-2].std(0)], f, 2)

        # dataSL = data[:, -2:]
        # scaler = preprocessing.StandardScaler().fit(data[:, :-2])
        # dataset_standardised = np.concatenate((scaler.transform(data[:, :-2]), dataSL), axis=1)
        
        # Filter on labels
        X_0 = data[data[:, -1]==0]
        X_1 = data[data[:, -1]==1]

        X_0LS = X_0[:, -2:]
        X_1LS = X_1[:, -2:]
        
        if X_0.size:
            scaler0 = preprocessing.StandardScaler().fit(X_0[:, :-2])
            X_0 = np.concatenate((scaler0.transform(X_0[:, :-2]), X_0LS), axis=1)

        if X_1.size:
            scaler1 = preprocessing.StandardScaler().fit(X_1[:, :-2])
            X_1 = np.concatenate((scaler1.transform(X_1[:, :-2]), X_1LS), axis=1)

        dataset_standardised = np.concatenate((X_0, X_1), axis=0)
        
        # Create pickle file with the input matrix
        with open(pickleFileName, 'wb') as f:
            pickle.dump(dataset_standardised, f, 2)

        return dataset_standardised



if __name__ == '__main__':
    # Import argument
    parser = argparse.ArgumentParser(description='Create the input matrix to feed the neural network')
    parser.add_argument('--path', type=str, default = './../openPoseDataset/', help='Enter folder root path of the position dataset')
    parser.add_argument('--discardLowBodyPart', type=bool, default = False, help='Flag to discard low body part')

    args = parser.parse_args()

    # Create vector
    process = Processing()
    dataset = process.createInputMatrix(args.path)

    # Standardise vector
    pickleFileName = args.path.split('/')[-2] + '.pkl'
    dataset = process.standardise(dataset, pickleFileName)
