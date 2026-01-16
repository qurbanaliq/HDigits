import gzip
import struct

import numpy as np

def get_images(filename):
    """
    loads mnist images dataset
    
    :param filename: file path to the images dataset
    """
    with gzip.open(filename,"rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # read and reshape the the data into 2D array
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        # convert to float and normalize the values to [0, 1]
        return images.astype(np.float32)/255.0

def get_labels(filename):
    """
    loads mnist labels dataset
    
    :param filename: file path to the labels dataset
    """
    with gzip.open(filename, "rb") as f:
        magic, num_lables = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels.astype(np.uint64)