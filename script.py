import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.optimizers import SGD
from keras.utils import to_categorical


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    # This function prints and plots the confusion matrix
    # Normalization can be applied by setting normalize=true

    plt.subplot(1, 3, 3)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if(normalize):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print "Normalized Confusion Matrix"
    else:
        print "Confusion Matrix, without Normalization"

    print cm

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

# ======= SETTING TRAINING AND TEST SETS =========

dataset_size = len(linear_images)
training_size = int((0.8) * dataset_size)
training_images = []
training_labels = []
test_images = []
test_labels = []

for i in range(0, training_size):
    training_images.append(linear_images[i])
    training_labels.append(labels[i])

for i in range(training_size, dataset_size):
    test_images.append(linear_images[i])
    test_labels.append(labels[i])

training_images = np.array(training_images)
training_labels = to_categorical(np.array(training_labels), num_classes=10)
test_images = np.array(test_images)
test_labels = to_categorical(np.array(test_labels), num_classes=10)

# ======= SETTING UP ANN MODEL  =========

model = Sequential()

model.add(Dense(units=784, activation='relu', input_shape=(784, )))
for i in range(0, 10):
    model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# Using 20% Dropout
keras.layers.Dropout(rate=0.2)

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# ======= TRAINING MODEL  =========

hist = model.fit(x=training_images, y=training_labels, epochs=500,
                 batch_size=512, verbose=1, validation_split=0.33)

# Alternatively, you can feed batches to your model manually:
# model.train_on_batch(x_batch, y_batch)

# ======= TESTING MODEL  =========

predictions = model.predict(x=test_images, batch_size=512, verbose=2)

# Evaluate your performance in one line:
loss_and_metrics = model.evaluate(
    x=test_images, y=test_labels, batch_size=512, verbose=2)

# ======= GENERATING METRICS  =========

print loss_and_metrics

y_pred = (predictions > 0.5)
cm = confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1))

cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_confusion_matrix(cm, cm_plot_labels, title="Confusion Matrix on Test Set")

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


print("SUCCESS!!")
