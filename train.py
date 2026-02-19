import os

import numpy as np

import model
import load_data
import predict

BASE_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_PATH, "MNIST")

def one_hot(labels, num_classes=10):
    """
    one-hot for labels
    
    :param labels: array of true labels
    :param num_classes: total number of classes
    """
    out = np.zeros((labels.size, num_classes))
    out[np.arange(labels.size), labels] = 1
    return out.T # (10, batch)

def trainModel():
    """
    trains the model on MNIST dataset
    """

    X_train = load_data.get_images(os.path.join(DATASET_PATH, "train-images-idx3-ubyte.gz"))
    y_train = load_data.get_labels(os.path.join(DATASET_PATH, "train-labels-idx1-ubyte.gz"))
    y_train_oh = one_hot(y_train)

    # convert dataset to [784, N]
    X_train = X_train.reshape(-1, 784).T

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
        y_train = y_train[perm]

        correct = 0 # number of correct predictions
        total_loss = 0 # total loss per epoch

        for i in range(0, N, batch_size):
            # pick the batch size chunk
            x = X_train[:, i:i + batch_size]
            y = y_train_oh[:, i:i + batch_size]
            labels = y_train[i:i + batch_size]

            # get the probabilities of the true classes and loss
            digitsModel.forward(x)
            loss = digitsModel.cross_entropy_loss(y, digitsModel.out)
            total_loss += loss * x.shape[1]

            # get the predicted digits
            preds = np.argmax(digitsModel.out, axis=0)

            # get the number of correct predictions
            correct += np.sum(preds == labels)

            # update the parameter in the backward pass
            digitsModel.backward(x, y, lr)
        
        print(
            f"Epoch {epoch + 1}: "
            f"Loss={total_loss/N} "
            f"Accuracy={correct/N*100:.2f}%"
        )
    
    # save the model
    path = os.path.join(BASE_PATH, "mnist_mlp.npz")
    digitsModel.save(path)
    print(f"model saved to '{path}'")
    
    return digitsModel

if __name__ == "__main__":
    # train the model
    _ = trainModel()
    predict.predict()
