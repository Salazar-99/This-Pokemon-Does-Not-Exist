import tensorflow as tf 
import matplotlib.pyplot as plt
from matplotlib import image

def plot_loss(history):
    loss = history.history['loss']
    epochs = [x+1 for x in range(len(loss))]
    plt.plot(epochs, loss, color="blue")
    plt.title('Training Loss vs. Epoch')
    plt.grid(alpha=0.5)
    plt.show()

def sample_from_vae(vae, n_samples):
    z = tf.random.normal([n_samples, vae.latent_dims])
    images = vae.decode(z)
    plt.imshow(images)
    return images

def sample_from_gan(gan, n_samples):
    z = tf.random.normal([n_samples, gan.coding_size])
    images = gan.generator(z)
    plt.imshow(images)
    return images