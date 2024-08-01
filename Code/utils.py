import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


def load_mnist():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Flatten images and normalize
    X_train = X_train.reshape(-1, 28 * 28).astype('float32') / 255
    X_test = X_test.reshape(-1, 28 * 28).astype('float32') / 255
    
    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))
    
    return X_train, y_train_one_hot, X_test, y_test_one_hot