import os

import numpy as np

from . import model
from . import load_data

DATASET_PATH = os.path.join(os.path.dirname(__file__), "MNIST")

def one_hot(labels, num_classes=10):
    """
    one-hot for labels
    
    :param labels: array of true labels
    :param num_classes: total number of classes
    """
    out = np.zeros((labels.size, num_classes))
    out[np.arange(labels.size), labels] = 1
    return out.T # (10, batch)

def accuracy(preds, labels):
    """
    returns the accuracy of the predictions
    
    :param preds: predictions made by the model
    :param labels: array of true labels
    """
    return np.mean(preds == labels) * 100

def trainModel():
    """
    trains the model on MNIST dataset
    """

    X_train = load_data.get_images(os.path.join(DATASET_PATH, "train-images-idx3-ubyte.gz")).T
    y_train = load_data.get_labels(os.path.join(DATASET_PATH, "train-labels-idx1-ubyte.gz"))
    y_train_oh = one_hot(y_train)

    digitsModel = model.HDigitsModel()

    epochs = 5
    batch_size = 64
    lr = 0.1 # learning rate
    N = X_train.shape[1] # number of training examples

    for epoch in range(epochs):
        # shuffle the training examples in each epoch
        perm = np.random.permutation(N)
        X_train = X_train[:, perm]
        y_train_oh = y_train_oh[:, perm]
        y_train = y_train[:, perm]

        correct = 0 # number of correct predictions
        total_loss = 0 # total loss per epoch

        for i in range(0, N, batch_size):
            # pick the batch size chunk
            x = X_train[:, i:i + batch_size]
            y = y_train_oh[:, i:i + batch_size]
            labels = y_train[:, i:i + batch_size]

            # get the probabilities of the true classes and loss
            probs = digitsModel.forward(x)
            loss = digitsModel.cross_entropy_loss(y, probs)
            total_loss += loss * x.shape[1]

            # get the predicted digits
            preds = np.argmax(probs, axis=0)

            # get the number of correct predictions
            correct += np.sum(preds == labels)
        
        print(
            f"Epoch {epoch + 1}:"
            f"Loss: {total_loss/N}"
            f"Accuracy: {correct/N*100:.2f}%"
        )
