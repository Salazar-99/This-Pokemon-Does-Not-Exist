from models import VAE
from data import fetch_data
from losses import vae_loss

#TODO: Add command line argument parsing

#Get data
data = fetch_data()

#Build model
vae = VAE(conv_layers=4, latent_dims=2)
vae.add_loss(vae_latent_loss)
vae.compile(loss=vae_loss, optimizer="adam")

#Train model
history = vae.fit(data, data, epochs=50, batch_size=32)

#TODO: Add diagnostics and sampling

#Plot diagnostics

#Plot samples