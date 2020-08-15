import tensorflow as tf
from matplotlib import image

#Return tf.dataset containing image arrays
def fetch_data():
    path = "/home/gerardo/Desktop/Projects/Datasets/Pokemon/images"
    images = []
    for file in os.listidr(path):
        filepath = path + "/" + file
        images.append(image.imread(filepath))
    dataset = tf.data.Dataset.from_tensor_slices(images)
    return dataset

#Normalize pixel values to range (0,1)
def transform_for_vae(dataset, batch_size):
    return dataset.map(lambda x: x/255).batch(batch_size).prefetch(1)

#Map pixel values to range (-1,1)
def transform_for_gan(dataset, batch_size):
    return dataset.map(lambda x: (x/255)*2-1).batch(batch_size).prefetch(1)