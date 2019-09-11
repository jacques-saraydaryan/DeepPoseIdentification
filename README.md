# DeepPoseIdentifcation

## Objective

Detect human positions on 2D camera using OpenPose. Classify people on images according to their degree of attention (focus or distract) though a neural network. The aim is to create a pipeline from jpeg images to be able to labelise images or video stream depending on people's attention for security purposes.

## Dataset

The dataset comes from two different sources: internet and own images. Different proportions of this two sources are taken for the training.

## OpenPose

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) represents the first real-time multi-person system to jointly detect human body, hand, facial and foot keypoints (17 in our case) on a single image.

## Neural Network

We are using Keras with tensorflow backend to build our deep learning algorithm.

### Choice of Neural network technolgy

TODO

### Choice of parameters

TODO

## Install

First let's get the [OpenPose Service](https://github.com/jacques-saraydaryan/ros-openpose) to perform the open pose detection.
Then clone the projet using:

```
$ git clone https://github.com/Pierre-Assemat/DeepPoseIdentification.git
```

## Contribution

Nina Tubau Ribera (Image Major)
Pierre Assemat (Robotic Major)

## TO DO

- [x] Create and organise dataset

- [x] Detect OpenPose positions from images (through Ros service)

- [x] Preprocess OpenPose data to create input matrix

- [x] Prototype classification with sklearn (machine learning)

- [x] Build a Neural Network using TFLearn

- [x] Train it --> Not efficient

- [x] Build a Neural Network using Keras

- [x] Train it --> Efficient

- [x] Connect the neural network to the processing chain

- [x] Create the prediction chain

- [x] Connect the video stream to the prediction chain

- [x] Improve the dataset using corner cases
