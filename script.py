# Written by:  Andrew Robbertz alrobbertz@wpi.edu
# Last Updated: 09/09/2018
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.optimizers import SGD
from keras.utils import to_categorical


def plot_metrics(con_matrix, classes, hist, title='Confusion Matrix', cmap=plt.cm.Blues):
    # This function prints and plots the confusion matrix
    # Normalization can be applied by setting normalize=true
    plt.subplot(1, 3, 3)
    plt.imshow(con_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = con_matrix.max() / 2
    for i, j in itertools.product(range(con_matrix.shape[0]), range(con_matrix.shape[1])):
        plt.text(j, i, con_matrix[i, j], horizontalalignment='center',
                 color='white' if con_matrix[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.subplot(1, 3, 1)
    plt.plot(hist.history['acc'], label='Training Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(hist.history['val_acc'], label='Validation Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.grid()
    plt.show()


def plot_fold_validation(scores):
    plt.subplot(1, 3, 1)
    plt.plot(scores[0], label='Training Accuracy')
    plt.plot(scores[1], label='Training Accuracy')
    plt.title('Accuracy Over Fold 1')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(scores[2], label='Training Accuracy')
    plt.plot(scores[3], label='Validation Accuracy')
    plt.title('Accuracy Over Fold 2')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(scores[4], label='Training Accuracy')
    plt.plot(scores[5], label='Validation Accuracy')
    plt.title('Accuracy Over Fold 3')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()


def plot_avg_validation(scores):
    avg_train_acc = []
    avg_val_acc = []
    for i in range(0, len(scores[0])):
        temp1 = [scores[0][i], scores[2][i], scores[4][i]]
        temp2 = [scores[1][i], scores[3][i], scores[5][i]]
        avg_train_acc.append(np.mean(temp1))
        avg_val_acc.append(np.mean(temp2))

    plt.plot(avg_train_acc, label='Training Accuracy')
    plt.plot(avg_val_acc, label='Validation Accuracy')
    plt.title('Avg Accuracy for Cross-Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()


def k_fold_validation(model, x_data, y_data, k=3, epochs=50, batch_size=32):
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Save the initial weights
    model.save_weights('weights.h5')

    kfold = StratifiedKFold(n_splits=k, shuffle=True)
    scores = []
    for index, (train_indicies, val_indicies) in enumerate(kfold.split(x_data, y_data)):
        # Reset the Weights for each fold
        model.load_weights('weights.h5')

        x_train, x_val = x_data[train_indicies], x_data[val_indicies]
        y_train, y_val = to_categorical(y_data[train_indicies], num_classes=10), to_categorical(
            y_data[val_indicies], num_classes=10)

        hist = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=epochs,
                         batch_size=batch_size, verbose=1)
        scores.append(hist.history['acc'])
        scores.append(hist.history['val_acc'])
    # plot_fold_validation(scores=scores)
    plot_avg_validation(scores=scores)
    return scores


def test_set__with_validation(x_data, y_data, epochs=50, batch_size=32):
    dataset_size = len(x_data)
    training_size = int((0.8) * dataset_size)
    training_images = []
    training_labels = []
    test_images = []
    test_labels = []
    # Divide the dataset into ttaining and test sets
    for i in range(0, training_size):
        training_images.append(x_data[i])
        training_labels.append(y_data[i])
    for i in range(training_size, dataset_size):
        test_images.append(x_data[i])
        test_labels.append(y_data[i])
    # Turn everything into np.array
    training_images = np.array(training_images)
    training_labels = to_categorical(np.array(training_labels), num_classes=10)
    test_images = np.array(test_images)
    test_labels = to_categorical(np.array(test_labels), num_classes=10)

    # Fit data to the model
    hist = model.fit(x=training_images, y=training_labels, epochs=epochs,
                     batch_size=batch_size, verbose=1, validation_split=0.33)
    # Make Prediction for Confusion Matrix
    predictions = model.predict(
        x=test_images, batch_size=batch_size, verbose=0)
    # Evaluate your performance in one line:
    loss_and_metrics = model.evaluate(
        x=test_images, y=test_labels, batch_size=batch_size, verbose=2)

    print loss_and_metrics
    y_pred = (predictions > 0.5)
    cm = confusion_matrix(test_labels.argmax(axis=1),
                          predictions.argmax(axis=1))
    cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plot_metrics(con_matrix=cm, classes=cm_plot_labels,
                 title="Confusion Matrix on Test Set", hist=hist)


# ======= LOADING AND LINEARLIZING IMAGES =========
images = np.load('images.npy')
labels = np.load('labels.npy')

linear_images = []
for i in range(0, 6500):
    linear_image = []
    for j in range(0, 28):
        for k in range(0, 28):
            linear_image.append(images[i][j][k])
    linear_images.append(linear_image)

# ======= GENERATING MODEL  =========

model = Sequential()
model.add(Dense(units=784, activation='relu', input_shape=(784, )))
for i in range(0, 10):
    model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# Using 20% Dropout
# keras.layers.Dropout(rate=0.2)

# Compie the model using the following metrics
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# test_set__with_validation(model=model, linear_images, labels, epochs = 20, batch_size = 512)

k_fold_validation(model=model, x_data=linear_images,
                  y_data=labels, k=3, epochs=500, batch_size=512)


print("SUCCESS!!")
