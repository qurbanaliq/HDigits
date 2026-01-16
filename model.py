import numpy as np

class HDigitsModel:
    def __int__(self, input_dim=784, hidden_dim=128, output_dim=10):
        
        self.W1 = self.xavier_init(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim, 1)

        self.W2 = self.xavier_init(hidden_dim, output_dim)
        self.b2 = np.zeros(output_dim, 1)

    def forward(self, x):
        """
        forward pass

        :param x: input batch
        """
        self.z1 = self.W1 @ x + self.b1
        self.h1 = self.relu(self.z1)

        self.z2 = self.W2 @ self.h1 + self.b2
        self.out = self.softmax(self.z2)
    
    def cross_entropy_loss(self, y_true, y_pred):
        """
        calculates the loss for the predicted values

        :param y_true: one-hot labels (10, batch)
        :param y_pred: predicted probabilities from softmax (10, batch)
        """
        # epsilon to prevent log(0)
        eps = 1e-12
        return -np.sum(y_true * np.log(y_pred + eps))/y_true.shape[1]

    def backward(self, x, y_true, lr):
        """
        backward propagation

        :param x: input batch (784, batch)
        :param y_true: one-hot labels (10, batch)
        :param lr: learning rate
        """
        batch_size = x.shape[1]

        # get the derivatives
        dz2 = (self.out - y_true) / batch_size
        dW2 = dz2 @ self.h1.T
        db2 = np.sum(dz2, axis=1, keepdims=True)

        dh1 = self.W2.T @ dz2
        dz1 = dh1 * self.relu_derivation(self.z1)
        dW1 = dz1 @ x.T
        db1 = np.sum(dz1, axis=1, keepdims=True)

        # update the parameters
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
    
    def predict(self, x):
        """
        predicts the value of given image data

        :param x: input image data
        """
        probs = self.forward(x)
        return np.argmax(probs, axis=0)

    @staticmethod
    def xavier_init(size_in, size_out):
        """
        Xavier initialization for weights
        
        :param size_in: input size of the weights
        :param size_out: output size of the weights
        """
        limit = np.sqrt(6/(size_in + size_out))
        return np.random.uniform(-limit, limit, (size_out, size_in))
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivation(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(z):
        z = z - np.max(z, axis=0, keepdims=True)
        e = np.exp(z)
        return e/np.sum(e, axis=0, keepdims=True)
    
