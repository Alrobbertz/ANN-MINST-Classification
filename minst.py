import numpy as np
import matplotlib.pyplot as plt


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical

from sklearn.model_selection import StratifiedKFold


def preprocessig():
    # Load in the stock dataset from keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Flatten the images into a 784 (28*28) long array
    x_train = [[pixel_val for row in image for pixel_val in row] for image in x_train]
    x_test = [[pixel_val for row in image for pixel_val in row] for image in x_test]

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    return x_train, x_test, y_train, y_test


def create_model(hidden_layers=10, activation_function="relu", dropout_rate=0.0):
    model = Sequential()
    # Add a dropout layer on the input between [0, 1] (0% dropout - 100% dropout)
    if dropout_rate > 0:
        model.add(Dropout(rate=dropout_rate, input_shape=(784,)))
    # Add an input layer that takes in whole image
    model.add(Dense(units=784, activation=activation_function, input_shape=(784,)))
    # Add hidden layers as specified
    for i in range(hidden_layers):
        model.add(Dense(units=50, activation=activation_function))
    # Add a single output layer with [0, 10] indicating what number the digit was
    model.add(Dense(units=10, activation='softmax'))

    # Print a summary of the model
    print model.summary()

    # Compile the model using the following metrics
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def k_fold_validation(model, x_data, y_data, k=3, epochs=50, batch_size=32):
    # Save the initial random weights
    model.save_weights('weights.h5')
    # Divide the dataset into folds
    kfold = StratifiedKFold(n_splits=k, shuffle=True)
    scores = []
    for index, (train_indicies, val_indicies) in enumerate(kfold.split(x_data, y_data)):
        print("Validation on Fold ", index)
        # Reset the Weights for each fold
        try:
            model.load_weights('weights.h5')
        except ValueError:
            print("No Weights to Load")
        # Load the Correct Training/ Validation Data
        x_train, x_val = x_data[train_indicies], x_data[val_indicies]
        y_train, y_val = to_categorical(y_data[train_indicies], num_classes=10), to_categorical(y_data[val_indicies], num_classes=10)
        # Fit data to model with validation data
        hist = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=epochs,
                         batch_size=batch_size, verbose=1)
        scores.append(hist.history['acc'])
        scores.append(hist.history['loss'])
        scores.append(hist.history['val_acc'])
        scores.append(hist.history['val_loss'])
        print("Training Accuracy: ", hist.history['acc'][-1])
        print("Validation Accuracy: ", hist.history['val_acc'][-1])

    return scores


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


def main():
    # Load the Data, Flatten it,
    x_train, y_train, y_train, y_test = preprocessig()
    print("Loaded Images Successfully")
    # Create and Compile your model
    model = create_model(hidden_layers=1, activation_function="relu", dropout_rate=0.1)
    # Run K-Fold validation on the model
    scores = k_fold_validation(model=model, x_data=x_train, y_data=y_train, k=3, epochs=50, batch_size=512)
    # For K-Fold Validation with Metrics for each Fold
    plot_fold_accuracy(scores=scores)
    plot_avg_accuracy(scores=scores)


if __name__ == "__main__":
    main()


print("YOUR PROGRAM FINISHED SUCCESS!!")
