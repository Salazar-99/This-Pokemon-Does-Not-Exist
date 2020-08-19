from models import GAN, train_gan
from data import fetch_data, transform_for_gan
from utils import plot_loss, sample_from_gan
import argparse

#Command line argument parsing
parser = argparse.ArgumentParser(description='Specify model hyperparameters')
parser.add_argument('--coding_size', type=int, help="Specify integer size of latent codings")
parser.add_argument('--epochs', type=int, help="Specify integer number of epochs")
args = parser.parse_args()

#Get data
batch_size = 32
raw_data = fetch_data()
data = transform_for_gan(raw_data, batch_size=batch_size)

#Build model
gan = GAN(coding_size=args.coding_size)

#Train model
train_gan(gan, data, batch_size, args.coding_size, args.epochs)

#Plot samples
sample_from_gan(gan, n_samples=5)

#Save model
gan.save('Models/GAN')