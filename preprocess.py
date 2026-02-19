from PIL import Image
import numpy as np

def preprocess_image(path):
    img = Image.open(path).convert("L")   # grayscale
    img = img.resize((28, 28))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = 1.0 - img
    return img
