from models import VAE, vae_loss , train_vae
from data import fetch_data_for_vae
from utils import plot_loss, sample_from_vae, upload_model
import tensorflow as tf
import argparse

#Command line argument parsing
parser = argparse.ArgumentParser(description='Specify model hyperparameters')
parser.add_argument('--conv_layers', type=int, help="Specify integer number of convolutional layers")
parser.add_argument('--latent_dims', type=int, help="Specify integer number of learned latent dimensions")
parser.add_argument('--epochs', type=int, help="Specify integer number of epochs")
parser.add_argument('--path', help="Path to dataset directory")
args = parser.parse_args()

#Get data
print("Fetching data...")
path = args.path
batch_size = 32
data = fetch_data_for_vae(path, batch_size)

#Build VAE
print("Building model...")
vae = VAE(conv_layers=args.conv_layers, latent_dims=args.latent_dims)

#Train VAE
print("Starting training...")
losses = train_vae(vae, data, batch_size, epochs=args.epochs)

#Plot diagnostics
plot_loss(losses)

#Plot some samples
sample_from_vae(vae)

#Upload trained model to S3 bucket
print("Would you like to save and upload the trained model? (Y/N): ")
choice = input()
if choice in ["Y", "Yes", "yes"]:
    vae.save('Models/VAE')
    upload_model(path_to_model='Models/VAE', object_name='vae')