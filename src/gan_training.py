from models import Generator, Discriminator, train_gan
from data import fetch_data_for_gan
from utils import plot_gan_losses, sample_from_gan, upload_model
import tensorflow as tf
import argparse

#Command line argument parsing
parser = argparse.ArgumentParser(description='Specify model hyperparameters')
parser.add_argument('--coding_size', type=int, help="Specify integer size of latent codings")
parser.add_argument('--conv_layers', type=int, help="Specify integer number of convolutional layers in Generator and Discriminator")
parser.add_argument('--epochs', type=int, help="Specify integer number of epochs")
parser.add_argument('--path', help="Path to dataset directory")
args = parser.parse_args()

#Get data
print("Fetching data...")
batch_size = 32
data = fetch_data_for_gan(path=args.path, batch_size=batch_size)

#Build model
print("Building model...")
generator = Generator(coding_size=args.coding_size)
discriminator = Discriminator(conv_layers=args.conv_layers)
gan = tf.keras.models.Sequential([generator, discriminator])
discriminator.compile(loss='binary_crossentropy', optimizer='rmsprop')
discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer='rmsprop')

#Train model
print("Starting training...")
losses = train_gan(gan, data, args.epochs)

#Plot losses
plot_gan_losses(losses)

#Plot samples
sample_from_gan(gan)

#Upload trained model to S3 bucket
print("Would you like to save and upload the trained model? (Y/N): ")
choice = input()
if choice in ["Y", "Yes", "yes"]:
    gan.save('Models/GAN')
    upload_model(path_to_model='Models/GAN', object_name='gan')