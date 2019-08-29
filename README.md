# DeepPoseIdentifcation

## Objective

Detect human positions on 2D camera using OpenPose. Classify people on images according to their degree of attention (focus or distract) though a neural network. The aim is to create a pipeline from jpeg images to be able to labelise images depending on people's attention for security purposes.   

## Dataset

The dataset comes from two different sources: internet and own images. Different proportions of this two sources are taken for the training.

## OpenPose

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) represents the first real-time multi-person system to jointly detect human body,hand, facial and foot keypoints (17 in our case) on single image.

## Neural Network

We are using TFLearn to construct our deepLearning algorithm.

## TO DO

[x] Create and organise dataset 
[x] Detect OpenPose positions from images (through Ros service)
[x] Preprocess OpenPose data to create input matrix 
[ ] Prototype classification with sklearn (machine learning)
[ ] Build a NN


## Contribution

Nina Tubau (Image Major)
Pierre Assemat (Robotic Major)
