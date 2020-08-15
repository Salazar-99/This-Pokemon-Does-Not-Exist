from models import VAE
from data import fetch_data
from losses import vae_loss
from utils import plot_loss, sample_from_vae

#TODO: Add command line argument parsing

#Get data
batch_size=32
raw_data = fetch_data()
data = transform_for_vae(raw_data, batch_size=batch_size)

#Build model
vae = VAE(conv_layers=4, latent_dims=2)
vae.add_loss(vae_latent_loss)
vae.compile(loss=vae_loss, optimizer="adam")

#Train model
history = vae.fit(data, data, epochs=50, batch_size=32)

#Plot diagnostics
plot_loss(history)

#Plot samples
sample_from_vae(vae, n_samples=5)