import tensorflow as tf 
import matplotlib.pyplot as plt
from matplotlib import image

def plot_loss(loss):
    epochs = [x+1 for x in range(len(loss))]
    plt.figure(figsize=(10,5))
    plt.plot(epochs, loss, color="blue", alpha=0.6)
    plt.title('Training Loss vs. Batch')
    plt.grid(alpha=0.5)
    plt.show()

#Generate n_samples images from trained VAE
def sample_from_vae(vae, n_samples):
    z = tf.random.normal([n_samples, vae.latent_dims])
    images = vae.decode(z)
    plot_images(images)
    return images

#Generate n_samples images from trained GAN
def sample_from_gan(gan, n_samples):
    z = tf.random.normal([n_samples, gan.coding_size])
    images = gan.generator(z)
    plot_images(images)
    return images

#TODO: Implement function for plotting multiple images on one figure
def plot_images(images):
    pass