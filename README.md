# ANN-MINST-Classification

Project Desctiption for CS 4341 Introduction to Artificial Intelligence

Professor: Neil T. Heffernan

## Project - 2 Artificial Neural Networks

# Primary Goal:

	In this project you will build an artificial neural network for categorizing images.
	You will write programs that take images like hand-written
	numbers and output what numbers the images represent.

# Data-set:
	MNIST is a data-set composed of handwritten numbers and their labels.
	It is a famous data-set that has been used for testing new machine-learning algorithm's performance. 
	Each MNIST image is a 28x28 grey-scale image. Data is provided as 28x28 matrices 
	containing numbers ranging from 0 (white pixels) to 255 (black pixels). 
	Labels for each image are also provided with integer values ranging from 0 to 9, 
	corresponding to the actual value in the image. You can download the whole  MNIST database 
	of handwritten digits (Links to an external site.)Links to an external site. by using the package
	 Keras in Python. We also provide you a smaller version of MNIST :images.npyPreview the document, 
	 labels.npyPreview the document. There are 6,500 images in our version of the database and 6,500 
	 corresponding labels. Since the original MNIST is very large, you are recommended to use the 
	 smaller version for this assignment, especially for Task 4.4

# Artificial Neural Network for classify the preprocessed data

	To implement an Artificial Neural Network, you should use a python package called Keras 
	implemented in Python3.
	You will build an ANN of three fully connected layers (input layer, hidden layer(s) and output layer).
	Then train and test your ANN on the MNIST dataset.
	You will use stochastic gradient descent to train your ANN. The loss function should be standard categorical 
	cross-entropy. The learning rate should be 0.001. Start by using 10 hidden layers, each with 50 nodes. 
	The batch size should be 512, with 500 epochs.
	Split the data into 20% test set, 20% validation set, and 60% training.
	Plot the accuracy of the training set and validation set over epochs. What conclusions can you draw about 
	your model based on the plot?
	Report the accuracy and the error and the confusion matrix on the test set.
	Add a 20% Dropout to the first layer of your model. Again, plot your training set and validation set over
	epochs. Compare your plot to the previous one, what conclusions can you draw about your model now?
	Again, Report the accuracy, the error, and confusion matrix on the test set.

# Cross validation

	You will implement a function of 3-fold cross validation. The function takes the model and dataset as 
	parameters, and returns the accuracies on training and validation sets for each fold.

# Evaluate Hyper-parameter configuration

	You will experiment with the hidden layers your ANN implemented in Task 1. Use the 3-fold cross validation 
	o evaluate the ANNs with 1, 2, 10 hidden layers. Report the best one and explain why it outperforms others.
	You will evaluate the model with batch size 32 and 512 by 3-fold cross validation. Report the best one and 
	explain the advantage and disadvantage of small batch size.
	Run an experiment of your own design and report on your findings.

