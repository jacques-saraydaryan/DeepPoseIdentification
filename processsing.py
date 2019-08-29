import json
import pickle
import glob
import argparse

# Run on Python 3 only
# the recursive option of the glob module is not available on Python 2

# Constants
LABEL = ['focus', 'distract']
SOURCE = ['google', 'perso']

# Import argument
parser = argparse.ArgumentParser(description='Create the input matrix to feed the neural network')
parser.add_argument('--path',type=str,default = './../openPoseDataset/',help='Enter folder root path of the position dataset')

def createInputMatrix(path):
    files = [f for f in glob.glob(path + "**/*.*",recursive=True)]
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
            for j in range(len(json_positions[i]['body_part'])):
                personList.append(json_positions[i]['body_part'][j]['x'])
                personList.append(json_positions[i]['body_part'][j]['y'])
                personList.append(json_positions[i]['body_part'][j]['confidence'])
            personList.append(LABEL.index(label))
            personList.append(SOURCE.index(source))
            data.append(personList)
    # Create pickle file with the input matrix
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.path

    createInputMatrix(path)
