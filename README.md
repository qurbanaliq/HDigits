# HDigits - MNIST From Scratch (NumPy)

A clean implementation of a Multilayer Perceptron (MLP) for handwritten digit classification using only NumPy.

This project trains a neural network on the MNIST dataset and allows prediction on custom images.

No deep learning frameworks were used — everything (forward pass, backpropagation, loss) is implemented from scratch.

---

## 📂 Project Structure

```text
HDigits/
│
├── MNIST/                     # Raw dataset files
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-labels-idx1-ubyte.gz
│
├── load_data.py               # MNIST loader
├── model.py                   # MLP model (forward, backward, save, load)
├── train.py                   # Training script
├── predict.py                 # Predict digit from custom image (random mnist test image by default)
├── preprocess.py              # Image preprocessing utilities
│
├── mnist_mlp.npz              # Saved trained model
├── sample.png                 # Example test image
│
└── README.md
```
---

## 🧠 Model Architecture

- Input Layer: 784 neurons (28x28 image flattened)
- Hidden Layer: 128 neurons
- Output Layer: 10 neurons (digits 0–9)
- Activation:
  - ReLU (hidden)
  - Softmax (output)
- Loss:
  - Cross-Entropy
- Optimizer:
  - Stochastic Gradient Descent

---

## 🚀 How To Train

Make sure the MNIST `.gz` files are inside the `MNIST/` folder.

Run:

python train.py

After training completes:

The model is automatically saved as mnist_mlp.npz in the main project folder

---

## 🔮 How To Predict

You can predict a digit from any image:

python predict.py path/to/image.png

Example:

python predict.py sample.png

---

## ⚠ Important: Image Requirements

The model was trained on MNIST images, which have:

- 28x28 resolution
- Grayscale
- White digit on black background
- Centered digit

If you create an image manually (e.g. in MS Paint):

The model inverts colors (during preprocessing to convert any image to 28x28) because:

MNIST format:
- Background = black (0)
- Digit = white (1)

Paint default:
- Background = white
- Digit = black

This inversion is handled inside preprocess.py.

If your model always predicts the same digit (e.g., always "8"), the issue is almost certainly incorrect preprocessing.

---

## 🧪 Preprocessing Steps

For any input image:

1. Convert to grayscale
2. Resize to 28×28
3. Normalize to [0,1]
4. Invert colors

---

## 💾 Saving and Loading the Model

Inside model.py:

model.save("mnist_mlp.npz")
model.load("mnist_mlp.npz")

This allows training once and predicting anytime later without retraining.

---

## 📚 What This Project Demonstrates

- Manual implementation of forward propagation
- Backpropagation derivation
- Cross-entropy loss with softmax
- Mini-batch gradient descent
- Dataset shuffling per epoch
- Numerical stability using epsilon
- Model serialization (save/load)
- Handling real-world preprocessing mismatches

---

## 🎯 Current Limitations

- Sensitive to digit position (MLP is not translation invariant)
- Works best on MNIST-style centered digits
- minimal file argument handling as it's not the purpose of this project
- No data augmentation

Future improvement:
- Add centering algorithm
- Improve robustness to foreground/background color inversion
- Implement a Convolutional Neural Network (CNN)

---

## 📌 Dependencies

- Python 3.x
- NumPy
- Pillow (for image preprocessing)
- Matplotlib

---

## 👨‍💻 Author Notes

This project was built as a deep dive into understanding the internal mechanics of neural networks.

The goal was not just accuracy — but structural understanding.

---

## 📊 Sample Accuracy

Training accuracy: ~97–98%  
Test accuracy: ~96–97%

(Depending on hyperparameters)

---

If you’re reading this — you now understand neural networks beyond the surface level.
