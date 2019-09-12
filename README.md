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


## Run

### Data preparation and training chain

First activate the service Openpose using ROS by running:

```
$ roslaunch openpose_ros_node serviceReadyTest.launch
```

To add more images in the dataset, you can rename the images using `training/utils.py`. 
(It can only be run with Python 3. Be aware, the rename function can overwrite images and loose some of them in the process. Use a mask to be sure to avoid any overwritting).

Then run the python (with Python 3) script `training/RosOpenPoseFiles.py` with 2 arguments:

- `--input`: Path to the folder containing the image dataset.

- `--output`: Path to the output folder where the json files will be created.

Once you get there, run the preprocess step `training/processing.py`  (with Python 3)  with 2 arguments:

- `--path`: Path to the folder containing all the joints position as json files.

- `--discardLowBodyPart`: Flag indicating if you want to discard all the lower joins from the features.

Then by running `training/training_bis.py` with the argument:

- `--path`: Path to the openPose dataset pickle file created by processing.py.

- `--epochNb`: The number of epoch you want your network to learn.

You will train your network. Once you get there, the model will be saved as `model.json`, `model.h5` and the best weights for loss optimization will be saved as `weights.best.hdf5`. 

### Display learning metrics with TensorBoard

After the learning step, if you want to gain some insight, just run: 

```
$ tensorboard --logdir="tensorboard/keras_model/[date of learning]"
```

And open your browser on `http://localhost:6006/`, you will be able to see all the metrics of the neural network learning.

### Run the prediction chain with testing data as input

Simply run the following command:

```
$ python predictions/prediction_bis.py
```

This requieres the file `data_test.pkl` as input. This file is created during the training step (within the `split_data methods`), once the full dataset is loaded and the separation between training/testing set is done.

### Run the prediction chain with Video stream as input

First activate the video stream ros service to send the stream on a channel using the command:

```
$ roslaunch openpose_ros_node videostream.launch
```

In the other side, run the python script using ROS to subscribe to the video stream with the command: 

```
$ rosrun openpose_ros_examples dataProcessingMainNode.py
```

If you want to display the video stream in real time, you can also run the following command:

```
rosrun rqt_image_view rqt_image_view
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