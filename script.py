# Written by:  Andrew Robbertz alrobbertz@wpi.edu
# Last Updated: 09/09/2018
import numpy as np
import math
import itertools
import keras
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import losses
from keras.optimizers import SGD
from keras.utils import to_categorical


def plot_metrics(con_matrix, classes, hist, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['acc'], label='Training Accuracy')
    plt.plot(hist.history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
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
    plt.show()


def plot_fold_accuracy(scores):
    plt.subplot(1, 3, 1)
    plt.plot(scores[0], label='Training Accuracy')
    plt.plot(scores[1], label='Training Accuracy')
    plt.title('Accuracy Over Fold 1')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(scores[2], label='Training Accuracy')
    plt.plot(scores[3], label='Validation Accuracy')
    plt.title('Accuracy Over Fold 2')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(scores[4], label='Training Accuracy')
    plt.plot(scores[5], label='Validation Accuracy')
    plt.title('Accuracy Over Fold 3')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()


def plot_avg_accuracy(scores):
    avg_train_acc = []
    avg_val_acc = []
    for i in range(0, len(scores[0])):
        temp1 = [scores[0][i], scores[4][i], scores[8][i]]
        temp2 = [scores[2][i], scores[6][i], scores[10][i]]
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


def test_scores(scores1, scores2):
    avg_val_acc1 = []
    for i in range(0, len(scores1[0])):
        temp = [scores1[2][i], scores1[6][i], scores1[10][i]]
        avg_val_acc1.append(np.mean(temp))
    avg_val_acc2 = []
    for i in range(0, len(scores2[0])):
        temp = [scores2[2][i], scores2[6][i], scores2[10][i]]
        avg_val_acc2.append(np.mean(temp))
    # Calculate the T-Statistic an P-Value
    statistic, pvalue = stats.ttest_ind(avg_val_acc1, avg_val_acc2)
    return statistic, pvalue


def k_fold_validation(model, x_data, y_data, k=3, epochs=50, batch_size=32):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    # Save the initial weights
    model.save_weights('weights.h5')
    # Divide the dataset into folds
    kfold = StratifiedKFold(n_splits=k, shuffle=True)
    scores = []
    for index, (train_indicies, val_indicies) in enumerate(kfold.split(x_data, y_data)):
        print "Validation on Fold ", index
        # Reset the Weights for each fold
        try:
            model.load_weights('weights.h5')
        except ValueError:
            print "No Weights to Load"
        # Load the Correct Training/ Validation Data
        x_train, x_val = x_data[train_indicies], x_data[val_indicies]
        y_train, y_val = to_categorical(y_data[train_indicies], num_classes=10), to_categorical(
            y_data[val_indicies], num_classes=10)
        # Fit data to model with validation data
        hist = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=epochs,
                         batch_size=batch_size, verbose=0)
        scores.append(hist.history['acc'])
        scores.append(hist.history['loss'])
        scores.append(hist.history['val_acc'])
        scores.append(hist.history['val_loss'])
        print "Training Accuracy: ", hist.history['acc'][len(
            hist.history['acc']) - 1]
        print "Validation Accuracy: ", hist.history['val_acc'][len(
            hist.history['val_acc']) - 1]

    return scores


def single_fold_validation(model, x_data, y_data, epochs=50, batch_size=32):
    dataset_size = len(x_data)
    training_size = int((0.8) * dataset_size)
    training_images = np.array(x_data[:training_size])
    training_labels = to_categorical(
        np.array(y_data[:training_size]), num_classes=10)
    test_images = np.array(x_data[training_size:])
    test_labels = to_categorical(
        np.array(y_data[training_size:]), num_classes=10)

    # Fit data to the model
    hist = model.fit(x=training_images, y=training_labels, epochs=epochs,
                     batch_size=batch_size, verbose=1, validation_split=0.33)
    # Make Prediction for Confusion Matrix
    predictions = model.predict(
        x=test_images, batch_size=batch_size, verbose=0)
    # Evaluate Test Performance:
    test_metrics = model.evaluate(
        x=test_images, y=test_labels, batch_size=batch_size, verbose=2)
    # Generate Confusion Matrix
    cm = confusion_matrix(test_labels.argmax(axis=1),
                          predictions.argmax(axis=1))
    return hist, cm, test_metrics


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


# ======= RUNNING TEST 1  =========

test_model_1 = Sequential()
# Using 20% Dropout
#test_model_1.add(Dropout(rate=0.2, input_shape=(784, )))
test_model_1.add(Dense(units=784, activation='relu', input_shape=(784, )))
for i in range(10):
    test_model_1.add(Dense(units=50, activation='relu'))
test_model_1.add(Dense(units=10, activation='softmax'))
#print test_model_1.summary()

# Compie the model using the following metrics
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
test_model_1.compile(loss='categorical_crossentropy',
                     optimizer=sgd,
                     metrics=['accuracy'])

# For Single-Fold Validation with Test Metrics and Confusion Matrix
hist1, cm1, test_metrics1 = single_fold_validation(
    model=test_model_1, x_data=linear_images, y_data=labels, epochs=50, batch_size=512)

# For K-Fold Validation with Metrics for each Fold
# scores1 = k_fold_validation(model=test_model_1, x_data=linear_images,
#                             y_data=labels, k=3, epochs=500, batch_size=512)

# ======= RUNNING TEST 2  =========

test_model_2 = Sequential()
# Using 20% Dropout
test_model_2.add(Dropout(rate=0.2, input_shape=(784, )))
test_model_2.add(Dense(units=784, activation='relu'))
for i in range(10):
    test_model_2.add(Dense(units=50, activation='relu'))
test_model_2.add(Dense(units=10, activation='softmax'))
#print test_model_2.summary()

# Compie the model using the following metrics
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
test_model_2.compile(loss='categorical_crossentropy',
                     optimizer=sgd,
                     metrics=['accuracy'])

# For Single-Fold Validation with Test Metrics and Confusion Matrix
hist2, cm2, test_metrics2 = single_fold_validation(
    model=test_model_2, x_data=linear_images, y_data=labels, epochs=50, batch_size=512)

# For K-Fold Validation with Metrics for each Fold
# scores2 = k_fold_validation(
#     model=test_model_2, x_data=linear_images, y_data=labels, k=3, epochs=500, batch_size=512)

# ======= PRINT METRICS =========

# Loss and Accuracy from Evaluating Test Set
print "Test 1 Loss and Accuracy", test_metrics1
print "Test 2 Loss and Accuracy", test_metrics2

# For Single-Fold Validation with Test Metrics and Confusion Matrix
statistic, pvalue = stats.ttest_ind(
    hist1.history['val_acc'], hist2.history['val_acc'])
print "T-Statistic and P-Value:", statistic, pvalue

# For K-Fold Validation with Metrics for each Fold
# statistic, pvalue = test_scores(scores1=scores1, scores2=scores2)
# print "T-Statistic and P-Value:", statistic, pvalue

# ======= PLOT METRICS FOR TEST 1 =========
cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# For Single-Fold Validation with Test Metrics and Confusion Matrix
plot_metrics(con_matrix=cm1, classes=cm_plot_labels,
             title="Confusion Matrix on Test Set", hist=hist1)

# For K-Fold Validation with Metrics for each Fold
# plot_fold_accuracy(scores=scores1)
# plot_avg_accuracy(scores=scores1)

# ======= PLOT METRICS FOR TEST 2 =========

# For Single-Fold Validation with Test Metrics and Confusion Matrix
plot_metrics(con_matrix=cm2, classes=cm_plot_labels,
             title="Confusion Matrix on Test Set", hist=hist2)

# For K-Fold Validation with Metrics for each Fold
# plot_fold_accuracy(scores=scores2)
# plot_avg_accuracy(scores=scores2)

print("SUCCESS!!")
