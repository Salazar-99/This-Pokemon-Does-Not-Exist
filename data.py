import tensorflow as tf
from matplotlib import image
from numpy import float32
import os

def fetch_data(path):
    """
    Fetch and return tf.Dataset object containing images

    Arguments:
        path (str) - Path to dataset directory

    Returns:
        tf.Dataset object containing complete dataset of images
    """
    images = []
    for file in os.listdir(path):
        filepath = path + "/" + file
        #float32 for compatability with TF ops default
        images.append(image.imread(filepath).astype(float32))
    dataset = tf.data.Dataset.from_tensor_slices(images)
    return dataset


def fetch_data_for_vae(path, batch_size):
    """
    Fetch and preprocess data used to train VAE
    
    Arguments:
        path (str) - Path to dataset directory
        batch_size (int) - Size of chunks of data to be processed at one time by the model

    Returns:
        tf.Dataset object of preprocessed and batched images
    """
    dataset = fetch_data(path)
    def preprocess(x):
        """
        Scale pixel values to range (0,1)
        """
        data = x/255
        return data
    return dataset.map(preprocess).batch(batch_size)


def fetch_data_for_gan(path, batch_size):
    """
    Fetch and preprocess data used to train GAN

    Arguments:
        path (str) - Path to dataset directory
        batch_size (int) - Size of chunks of data to be processed at one time by the model

    Returns:
        tf.Dataset object of preprocessed and batched images
    """
    dataset = fetch_data(path)
    def preprocess(x):
        """
        Scale pixel values to range (-1,1)
        """
        data = (x/255)*2-1
        return data
    return dataset.map(preprocess).batch(batch_size)