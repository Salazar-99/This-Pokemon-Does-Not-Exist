from models import Generator, Discriminator, train_gan
from data import fetch_data_for_gan
from utils import plot_loss, sample_from_gan, upload_model
import argparse

#Command line argument parsing
parser = argparse.ArgumentParser(description='Specify model hyperparameters')
parser.add_argument('--coding_size', type=int, help="Specify integer size of latent codings")
parser.add_argument('--conv_layers', type=int, help="Specify integer number of convolutional layers in Generator and Discriminator")
parser.add_argument('--epochs', type=int, help="Specify integer number of epochs")
args = parser.parse_args()

#Get data
batch_size = 32
data = fetch_data_for_gan(batch_size=batch_size)

#Build model
generator = Generator(coding_size=args.coding_size, conv_layers=args.conv_layers)
discriminator = Discriminator(conv_layers=args.conv_layers)
gan = tf.keras.models.Sequential([generator, discriminator])
discriminator.compile(loss='binary_crossentropy', optimizer='rmsprop')
discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer='rmsprop')

#Train model
losses = train_gan(gan, data, batch_size, args.coding_size, args.epochs)

#Plot losses
plot_loss(losses['d_loss']) #Discriminator loss
plot_loss(losses['g_loss']) #Generator loss

#Plot samples
sample_from_gan(gan, n_samples=5)

#Upload trained model to S3 bucket
print("Would you like to save and upload the trained model? (Y/N): ")
choice = input()
if choice in ["Y", "Yes", "yes"]:
    tf.saved_model.save(vae, 'Models/GAN')
    upload_model(path_to_model='Models/GAN', object_name='gan')