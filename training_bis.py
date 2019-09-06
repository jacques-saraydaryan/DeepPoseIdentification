import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.constraints import unit_norm
from keras.callbacks import TensorBoard


class Training():

	def __init__(self, name_pickle=None, data=None):
		if name_pickle is not None :
			dataset = pd.read_pickle(name_pickle)
			data = pd.DataFrame(dataset)
		elif data is None:
			raise Exception("No input data found")

		self.data = data.sample(frac=1)


	def split_data(self, nb_features=54):
		# Creating input features and target variables
		X = self.data.iloc[:, :nb_features]
		y = self.data.iloc[:, -1]
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)


	def buildDNN(self, hidden_layers=2):
		self.classifier = Sequential()

		# First Layer
		self.classifier.add(Dense(54, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu',kernel_constraint=unit_norm(), kernel_initializer='random_normal', input_dim=54))
		self.classifier.add(Dropout(0.2))

		# Hidden Layer(s)
		for nbLayer in range(hidden_layers - 1):
			self.classifier.add(Dense(54, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu', kernel_initializer='random_normal'))
			self.classifier.add(Dropout(0.2))
		
		# Output Layer
		self.classifier.add(Dense(2, activation='sigmoid', kernel_initializer='random_normal'))
		
		# Compiling the neural network
		self.classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics =['accuracy'])


	def train(self, epochNb=300):
		logdir = "tensorboard/keras_model/" + datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard_callback = TensorBoard(log_dir=logdir)

		# Fitting the data to the training dataset
		self.history = self.classifier.fit(self.X_train, self.y_train, validation_split=0.33, batch_size=10, epochs=epochNb)

		# Save model as json file
		model_json = self.classifier.to_json()
		with open("model.json", "w") as f:
			f.write(model_json)
			f.close()
		
		# serialize weights to HDF5
		self.classifier.save_weights("model.h5")
		print("\nModel train and saved as 'model.json' and 'model.h5'")


	def plot_acc(self, all=None):
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
		# summarize history for loss
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()


	def plot_all(self):
		plt.figure()
		plt.subplot(1,2,1)
		self.plot_acc(True)
		plt.subplot(1,2,2)
		self.plot_loss()


if __name__ == '__main__':
    # Get argument parser
    parser = argparse.ArgumentParser(description='Chain of focus detection using human pose detection')
    parser.add_argument('--path', type=str, default='../openPoseDataset/', help='Path to input json dataset')
    parser.add_argument('--epochNb', type=int, default=500, help='Number of epoch wanted to train the NN')
    args = parser.parse_args()

    training = Training(name_pickle='openPoseDataset.pkl')
    training.split_data()
    training.buildDNN()
    training.train(args.epochNb)
    training.plot_all()


