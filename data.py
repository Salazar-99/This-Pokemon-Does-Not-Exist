import tensorflow as tf
from matplotlib import image
import os
from numpy import float32

#Return tf.dataset containing image arrays
def fetch_data():
    path = "/home/gerardo/Desktop/Projects/Datasets/Pokemon/images"
    images = []
    for file in os.listdir(path):
        filepath = path + "/" + file
        images.append(image.imread(filepath).astype(float32))
    dataset = tf.data.Dataset.from_tensor_slices(images)
    return dataset


def fetch_data_for_vae(batch_size):
    dataset = fetch_data()
    def preprocess(x):
        #Normalize pixel values to range (0,1)
        data = x/255
        #Return (input, target) which are the same image
        return data
    return dataset.map(preprocess).batch(batch_size)


def fetch_data_for_gan(batch_size):
    dataset = fetch_data()
    def preprocess(x):
        #Scale pixel values to range (-1,1)
        data = (x/255)*2-1
        #Return (input, target) which are the same image
        return data
    return dataset.map(preprocess).batch(batch_size)