from models import VAE
from data import fetch_data
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

#Build model
vae = VAE(args.conv_layers, args.latent_dims)
vae.add_loss(vae_latent_loss)
vae.compile(loss=vae_loss, optimizer="adam")

#Train model
history = vae.fit(data, data, epochs=args.epochs, batch_size=32)

#Plot diagnostics
plot_loss(history)

#Plot samples
sample_from_vae(vae, n_samples=5)

#Save model
vae.save('Models/VAE')