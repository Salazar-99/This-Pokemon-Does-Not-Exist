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

#Generate 5 images from trained VAE
def sample_from_vae(vae):
    z = tf.random.normal([5, vae.latent_dims])
    images = vae.decode(z)
    plot_images(images)
    return images

#Generate 5 images from trained GAN
def sample_from_gan(gan):
    z = tf.random.normal([5, gan.coding_size])
    images = gan.generator(z)
    plot_images(images)
    return images

def plot_images(images):
    #1 row, 5 columns
    columns = 5
    rows = 1
    fig=plt.figure(figsize=(10, 20))
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
    plt.show()