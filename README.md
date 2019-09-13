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

We started using TFLearn. However, because it is less handy than Keras, we switched to Keras. Keras is a high-level neural network API, written in Python and running on top of TensorFlow in our case. The advantage is that it allows an easy and fast prototyping. 

In our specific case, the aim is to classify in two caterories: focus and distract. It is actually a binary classification. Because it is a short term project, we decided to do a prototype, we chose a perceptron. 

### Choice of parameters

We chose the parameter of the network in a way that the loss is minimal and the accuracy is maximal. We can tweak the number of fully-connected layers of the network, number of neurons, activation functions, the optimiser, number of epochs.

- Number of layers

In our case the number of layers is 5. This choice is a tradeoff between overfitting and underfitting. It was made using the loss metric with Tensorboard. It is a parameter that can be changed when running `training.py`.

- Number of neurons

The number of neurons is set to 54. Considering that the number of features is also 54, we didn't want to loose any information by decreasing the number of neurons. After several tests, we noticed that 54 neurons for each layer is the most accurate.

- Activation function 

The activation function defines the output of a node given an input or set of inputs. They help neural networks (NN) introduce non-linearity. This means that the NN can successfully approximate functions which does not follow linearity or predict the class of a function which is divided by a decision boundary which is not linear. In our case, we chose `tanh`. It is very similar to sigmoid (suitable for classification) but has a stronger gradient meaning that the derivates are steeper and converge quicker.

As a last activation function, we use `sigmoid`. It restricts the output value between 0 and 1 which for us indicates the label of output.

- Optimiser

The optimiser is used in NN to produce better and faster results when updating the parameters, weights and biases. In our case, we used an Adaptive Moment Estimation (Adam). It computes adaptive learning rates for each parameter. Adam works well in practice, it converges very fast. It also rectifies problems such as vanishing learning rate, slow convergence or high variance in the parameter updates. 

- Number of epochs 

The number of epochs is a hyperparameter that defines the number of times that the learning algorithm will work trough the entire dataset. We set this number to 1000 using Tensorboard and looking when the metrics converge. 

- Metrics

The metrics are usefull to evaluate the network. In this case we are optimising on the loss function. More precisely, we use the binary cross entropy. It is suitable for classification problems such as ours.

- Problem: Overfitting

One of the main issues when training is overfitting. We detected that it was happening looking at the increase in the loss curve of the validation set over-time. 

Overfitting is the fact that the learning is too close to a particular set of data, even learning some noise. To avoid overfitting, we used classic tricks such as adding Dropout layers with a rate of 0.08 input units to 0, meaning that this units will sudently be ignored. By using fully-connected layers, neuurons develop co-dependency amongst each other during training leading to over-fitting.

Also, we added a regularisation term of 0.0007. Regularisation discourages learning a more complex or flexible model. Thus it will reduce overfitting.

Setting all this parameters, we achieve the following results: 

<img src="assets/result.png" alt="Results" width="550" style="margin: 20px 50%; transform:translateX(-50%)"/>



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
