import argparse

from processsing import Processing
from training import Training


if __name__ == '__main__':
    # Get argument parser
    parser = argparse.ArgumentParser(description='Chain of focus detection using human pose detection')
    parser.add_argument('--path', type=str, default='../openPoseDataset/', help='Flag to skip pose detection step')

    ## Start detection chain for training
    args = parser.parse_args()
    pathOutput = args.path

    # Concat all the positions data into a single array and save it as pickle file
    process = Processing()
    data = process.createInputMatrix(pathOutput)

    # Construct the Neural Network classifier and start the learning phase
    training = Training(data)
    net = training.buildNN(5)

    # Train the Neural Network
    training.train(net)
