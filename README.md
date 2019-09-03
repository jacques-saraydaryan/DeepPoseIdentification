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

- [x] Create and organise dataset

- [x] Detect OpenPose positions from images (through Ros service)

- [x] Preprocess OpenPose data to create input matrix

- [x] Prototype classification with sklearn (machine learning)

- [x] Build a Neural Network

- [ ] Train it

## Install

First let's get the [OpenPose Service](https://github.com/jacques-saraydaryan/ros-openpose) to perform pose detection.
Then just clone the projet using:

```
$ git clone https://github.com/Pierre-Assemat/DeepPoseIdentification.git
```

## Run

First activate the service Openpose using ROS by running:

```
$ roslaunch openpose_ros_node serviceReadyTest.launch
```

Once you get there, run the python script `RosOpenPoseFiles.py` with 2 arguments:

- `pathInput`: Path to folder containing the image dataset.

- `pathOutput`: Path to output folder where the json files will be created.

# Display metrics with TensorBoard

Once the training step is done, run the following command where `path/to/tfevents` is `tensorboard/dense_model` in our case:

```
$ tensorboard --logdir='path/to/tfevents'
```

Then open your browser on `localhost:6006`

## Contribution

Nina Tubau (Image Major)
Pierre Assemat (Robotic Major)
