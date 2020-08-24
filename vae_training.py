from models import VAE
from data import fetch_data, transform_for_vae
from losses import vae_loss
from utils import plot_loss, sample_from_vae
import argparse

#Command line argument parsing
parser = argparse.ArgumentParser(description='Specify model hyperparameters')
parser.add_argument('--conv_layers', type=int, help="Specify integer number of convolutional layers")
parser.add_argument('--latent_dims', type=int, help="Specify integer number of learned latent dimensions")
parser.add_argument('--epochs', type=int, help="Specify integer number of epochs")
args = parser.parse_args()

#Get data
batch_size = 32
raw_data = fetch_data()
data = transform_for_vae(raw_data, batch_size=batch_size)

#Build VAE
vae = VAE(conv_layers=args.conv_layers, latent_dims=args.latent_dims)

#Train VAE
losses = train_vae(vae, dataset, batch_size, epochs=args.epochs)

#Plot diagnostics
plot_loss(losses)

#Plot samples
sample_from_vae(vae, n_samples=5)

#Save model
vae.save('Models/VAE')