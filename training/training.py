import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

from keras import Sequential, optimizers
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.constraints import unit_norm
from keras.callbacks import TensorBoard, ModelCheckpoint


class Training():

	def __init__(self, name_pickle):
		dataset = pd.read_pickle(name_pickle)
		data = pd.DataFrame(dataset)
		self.data = data.sample(frac=1)


	def split_data(self, nb_features=54):
		'''Split dataset into training and testing data
		Input: dataset, number of features to take into account
		'''
		# Creating input features and target variables
		X = self.data.iloc[:, :nb_features]
		y = self.data.iloc[:, -1]

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

		# Create pickle file with the input matrix
		with open('data_test.pkl', 'wb') as f:
			pickle.dump([self.X_test, self.y_test] , f, 2)


	def buildDNN(self, hidden_layers=5):
		'''Building the architecture of the network (perceptron)
		Input: number of hidden layers
		'''
		self.classifier = Sequential()

		# First Layer
		self.classifier.add(Dense(54, kernel_regularizer=l2(0.0007), bias_regularizer=l2(0.0007), activation='tanh', kernel_constraint=unit_norm(), kernel_initializer='random_normal', input_dim=54))
		self.classifier.add(Dropout(0.08))

		# Hidden Layer(s)
		for nbLayer in range(hidden_layers - 1):
			self.classifier.add(Dense(54, kernel_regularizer=l2(0.0007), bias_regularizer=l2(0.0007), activation='tanh', kernel_initializer='random_normal'))
			self.classifier.add(Dropout(0.08))

		# Output Layer
		self.classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

		try:
			# load weights
			self.classifier.load_weights("weights.best.hdf5")
		except e:
			pass

		# Optimizer
		adam = optimizers.Adam(lr=0.001)

		# Compiling the neural network
		self.classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics =['accuracy'])


	def train(self, epochNb=1000):
		'''Train the network
		Input: Number of epochs
		Output: Save a json and h5 file with the weights
		'''
		logdir = "tensorboard/keras_model/" + datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard_callback = TensorBoard(log_dir=logdir)

		# checkpoint
		filepath="weights.best.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]

		# Fitting the data to the training dataset
		self.history = self.classifier.fit(self.X_train, self.y_train, validation_split=0.20, callbacks=callbacks_list, batch_size=32, epochs=epochNb)

		# Save model as json file
		model_json = self.classifier.to_json()
		with open("model.json", "w") as f:
			f.write(model_json)
			f.close()

		# serialize weights to HDF5
		self.classifier.save_weights("model.h5")
		print("\nModel trained and saved as 'model.json', 'model.h5' and 'weights.best.hdf5'")


	def plot_acc(self, all=None):
		'''Plot of the accuracy after training
		'''
		# summarize history for accuracy
		plt.plot(self.history.history['acc'])
		plt.plot(self.history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')

		if not all:
			plt.show()


	def plot_loss(self):
		'''Plot of the loss after training
		'''
		# summarize history for loss
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()


	def plot_all(self):
		'''Plot of the accuracy and loss (combination of plot_loss and plot_acc) after training
		'''
		plt.figure()
		plt.subplot(1,2,1)
		self.plot_acc(True)
		plt.subplot(1,2,2)
		self.plot_loss()


if __name__ == '__main__':
    # Get argument parser
    parser = argparse.ArgumentParser(description='Chain of focus detection using human pose detection')
    parser.add_argument('--path', type=str, default="./openPoseDataset.pkl", help='Path to input dataset pickle file')
    parser.add_argument('--epochNb', type=int, default=1000, help='Number of epoch wanted to train the NN')
    parser.add_argument('--layerNb', type=int, default=5, help='Number of hidden layers')
    args = parser.parse_args()

    training = Training(args.path)
    training.split_data()
    training.buildDNN(args.layerNb)
    training.train(args.epochNb)
    training.plot_all()

    #print(accuracy_score(training.y_test, training.classifier.predict_classes(training.X_test)))
