import matplotlib.pyplot as plt
from matplotlib import image
import tensorflow as tf
import boto3
import os

def plot_loss(loss):
    """
    Plot loss as a function of batch. Used for VAE.

    Arguments:
        loss (List[float]) - list of losses of length batch_size * epochs
    """
    batch = [x+1 for x in range(len(loss))]
    plt.figure(figsize=(10,5))
    plt.plot(batch, loss, color="blue", alpha=0.6)
    plt.title('Training Loss vs. Batch')
    plt.grid(alpha=0.5)
    plt.show()

def plot_gan_losses(losses):
    """
    Plot Generator and Discriminator losses as a function of batch. Used for GAN.

    Arguments:
        lossses (dict) - Dictionary containing 'd_loss' and 'g_loss' which are lists of length (batch_size * epochs)
    """
    d_loss = losses['d_loss']
    g_loss = losses['g_loss']
    batch = [x+1 for x in range(len(d_loss))]
    plt.figure(figsize=(10,5))
    plt.plot(batch, d_loss, alpha=0.6, color="blue", label="Discriminator")
    plt.plot(batch, g_loss, alpha =0.6, color="red", label="Generator")
    plt.title('Gan Losses vs. Batch')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

def sample_from_vae(vae):
    """
    Generate and plot 5 samples from VAE.

    Arguments:
        vae (VAE) - Trained VAE
    """
    z = tf.random.normal([5, vae.latent_dims])
    images = vae.decode(z)
    plot_images(images)
    return images

def sample_from_gan(gan):
    """
    Generate and plot 5 samples from GAN.

    Arguments:
        gan (GAN) - Trained GAN
    """
    generator, _ = gan.layers
    z = tf.random.normal([5, generator.coding_size])
    images = generator(z)
    plot_images(images)
    return images

def plot_images(images):
    """
    Helper function for plotting samples from model:

    Arguments:
        images (tf.Tensor) - Tensor of numpy arrays generated by a model

    Notes:
        The number of samples/plots is currently hardcoded to 5
    """
    columns = 5
    rows = 1
    fig = plt.figure(figsize=(10, 20))
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
    plt.show()

def upload_model(path_to_model, object_name, bucket='pokemon-dne'):
    """
    Uploads saved model to S3 bucket.

    Arguments:
        path_to_model (str) - Path to saved Tensorflow model directory
        bucket (str) - Name of S3 bucket to store model in
        object_name (str) - Name of the model once it's uploaded to S3

    Notes:
        In order to use boto3, AWS credentials must be configured on host machine.
        In this project this is accomplished by creating a .env file containing
        the nevessary credentials and running the configure_aws.sh script to 
        write them in the correct format to their default location.
    """
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(path_to_model):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, path_to_model)
            s3_path = os.path.join(object_name, relative_path)
            s3_client.upload_file(local_path, bucket, s3_path)