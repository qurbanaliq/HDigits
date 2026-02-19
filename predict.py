import model
from preprocess import preprocess_image
import load_data

import os
import sys
import random

import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_PATH, "MNIST")

def accuracy(preds, labels):
    """
    returns the accuracy of the predictions
    
    :param preds: predictions made by the model
    :param labels: array of true labels
    """
    return np.mean(preds == labels) * 100

def predict(image_path=None):
    """makes a prediction on randomly selected image from mnist test dataset
    """
    model_path = os.path.join(BASE_PATH, "mnist_mlp.npz")
    if not os.path.exists(model_path):
        print(f"Could not find saved model at {model_path}")
        return
    trainedModel = model.HDigitsModel.load(model_path)
    if image_path is None:
        X_test_flat = load_data.get_images(os.path.join(DATASET_PATH, "t10k-images-idx3-ubyte.gz"))
        y_test = load_data.get_labels(os.path.join(DATASET_PATH, "t10k-labels-idx1-ubyte.gz"))

        # convert images into [784, batch]
        X_test = X_test_flat.reshape(-1, 784).T

        # make the predictions
        test_preds = trainedModel.predict(X_test)
        
        # get the accuracy
        test_accuracy = accuracy(test_preds, y_test)

        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        index = random.randrange(10000)
        img = X_test_flat[index].reshape(28, 28)
        pred = test_preds[index]
    
    else:
        if not os.path.exists(image_path):
            print(f"Couldn't find the specified image: {image_path}")
            return
        img = preprocess_image(image_path) # get 28x28 image
        pred = trainedModel.predict(img.reshape(784, 1))

    plt.imshow(img, cmap="gray")
    plt.title(f"predicted: {pred}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    args = None
    if len(sys.argv) > 1:
        args = sys.argv[1]
    predict(args)
