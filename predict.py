from model import HDigitsModel
from preprocess import preprocess_image

import os

DIR = os.path.dirname(__file__)
saved_model = os.path.join(DIR, "mnist_mlp.npz")

model = HDigitsModel.load(saved_model)

x = preprocess_image(os.path.join(DIR, "sample.png"))
pred = model.predict(x.reshape(784, 1))

# show prediction on the UI
import matplotlib.pyplot as plt

plt.imshow(x, cmap="gray")
plt.title(f"predicted: {pred}")
plt.axis("off")
plt.show()

print("Predicted digit:", pred)
