import tensorflow as tf
from math import pi

PI = tf.constant(pi)

class VAE(tf.keras.Model):
    """
    Defines a Convolutional Variational Autoencoder with user-defined hyperparameters

    Arguments:
        conv_layers (int) - Number of convolutional layers in the Encoder. The Decoder 
            has 2 upscaling Transposed Convolution layers and then 2-conv_layers convolutional layers.
        latent_dims (int) - There are 2*latent_dims number of hidden units in the last (dense) layer 
            of the encoder. This represents the number of dimensions of the target latent normal distribution
    """
    def __init__(self, conv_layers, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.conv_layers = conv_layers
        
        #Build encoder
        self.encoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(120,120,3))])
        for i in range(conv_layers):
            filters = self.get_filters(i)
            self.encoder.add(tf.keras.layers.Conv2D(
                filters=filters, kernel_size=3, strides=(2,2), padding='same', activation='relu'))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(2*latent_dims))
        
        #Build decoder
        self.decoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dims,))])
        self.decoder.add(tf.keras.layers.Dense(units=60*60*32, activation='relu'))
        self.decoder.add(tf.keras.layers.Reshape(target_shape=(60, 60, 32)))
        #Upscaling layers
        for i in range(1):
            filters = self.get_filters(conv_layers-1-i)
            self.decoder.add(tf.keras.layers.Conv2DTranspose(
                filters=filters, kernel_size=(3,3), strides=2, padding='same', activation='relu'))
        #Remaining conv layers
        for i in range(1, conv_layers):
            filters = self.get_filters(conv_layers-1-i)
            self.decoder.add(tf.keras.layers.Conv2DTranspose(
                filters=filters, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid'))
    
    #Computing number of filters as a function of layer depth
    def get_filters(self, i):
        return 2**(i+4)

    #Utility for generating new data
    def decode(self, z):
        return self.decoder(z)
    
    #TODO: Make note in paper about why we do the split and the dense layer has 2xlatent_dims
    def encode(self, inputs):
        mean, logvar = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        return mean, logvar

    #Reparameterization trick
    def reparameterize(self, mean, logvar):
        noise = tf.random.normal(shape=mean.shape)
        return mean + noise * tf.exp(logvar * .5)

def train_vae(vae, dataset, batch_size, epochs, lr=1e-4):
    """
    Train VAE and return list of losses.

    Arguments:
        vae (tf.Model) - VAE to be trained
        dataset (tf.Dataset) - Datset containing all preprocessed and batched images
        batch_size (int) - Number of data points per batch
        epochs (int) - Total number of epochs to train for
        lr (float) - learning rate for Adam optimizer, defaults to 1e-4

    Returns:
        losses - List containing loss for each batch
    """
    losses = []
    optimizer = tf.keras.optimizers.Adam(lr)
    for epoch in range(epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                loss = vae_loss(vae, batch)
                losses.append(loss)
            gradients = tape.gradient(loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        print(f"Epoch: {epoch+1}, Loss: {losses[-1]}")
    return losses

@tf.function()
def vae_loss(model, input):
    """
    Loss function for Variational Autoencoder.

    Arguments:
        model (tf.Model) - VAE to be trained
        input (tf.Tensor) - Batched data points as input

    Returns:
        Sum of reconstruction loss and KL-divergence averaged over batched input
    """
    mean, logvar = model.encode(input)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=input)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def log_normal_pdf(sample, mean, log_var):
    """
    Helper function for vae_loss.
    Compute the the value of log-normal distribution pdf at the sample.

    Arguments:
        sample (np.array) - value at which the 
        mean (float) - mean of target log-normal distribution
        log_var (float) - log variance of target log-normal distribution

    Return:
        value of log-normal pdf parameterized by mean and log_var at the sample value
    """
    log_2pi = tf.math.log(2. * PI)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-log_var) + log_var + log_2pi), axis=1)

class Generator(tf.keras.Model):
    """
    Generator for a Deep Convolutional Generative Adversarial Network.
    Takes in a random vector of length coding_size and produces an image with dimensions (120,120,3).

    Arguments:
        coding_size (int) - Size of the random vector input
    """
    def __init__(self, coding_size):
        super().__init__()
        self.coding_size = coding_size
        self.input_layer = tf.keras.layers.Dense(15*15*512, input_shape=(self.coding_size,))
        self.reshape_layers = [
            tf.keras.layers.Reshape([15*15*512]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((15,15,512))
        ]
        self.conv_layers = []
        for i in reversed(range(1, 3)):
            filters = self.get_filters(i)
            self.conv_layers.append(tf.keras.layers.Conv2DTranspose(
                filters=filters, kernel_size=(3,3), strides=2, padding='same', activation='relu'))
            self.conv_layers.append(tf.keras.layers.BatchNormalization())
        self.conv_layers.append(tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='tanh'))

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.reshape_layers:
            x = layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        return x
    
    def get_filters(self, i):
        return 2**(5+i)

class Discriminator(tf.keras.Model):
    """
    Discriminator for a Deep Convolutional Generative Adversarial Network.
    Takes in an image with dimensions (120,120,3) and outputs the probability that it belongs to the original dataset.

    Arguments:
        conv_layers (int) - Number of 2D Convolutional layers in the network
    """
    def __init__(self, conv_layers):
        super().__init__()
        #First conv layer, seeding input shape
        self.input_layer = tf.keras.layers.Conv2D(
                filters=2**6, kernel_size=3, strides=(2,2), padding='same', 
                activation=tf.keras.layers.LeakyReLU(), input_shape=(120,120,3))
        #Remaining conv layers
        self.conv_layers = []
        for i in range(2, conv_layers):
            filters = self.get_filters(i)
            self.conv_layers.append(tf.keras.layers.Conv2D(
                filters=filters, kernel_size=3, strides=(2,2), padding='same', activation=tf.keras.layers.LeakyReLU()))
        #Final classifier layers
        self.classifier_layers = [
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    
    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.classifier_layers:
            x = layer(x)
        return x

    def get_filters(self, i):
        2**(5+i)

def train_gan(gan, dataset, epochs):
    """
    Train a Generative Adversarial Network and return batch losses

    Arguments:
        gan (tf.keras.Model) - Complete GAN model, consisting of a Generator and Discriminator
            stacked as a Sequential model
        dataset (tf.Dataset) - Batched and preprocessed set of images to train on
        epochs (int) - Number of times to train over the entire dataset

    Returns:
        losses (dict) - Dictionary containing lists of generator and discriminator losses at each batch
    
    Notes:
        batch_size computed inside loop to handle the possibility of a final batch of different length
    """
    generator, discriminator = gan.layers
    coding_size = generator.coding_size
    discriminator_loss = []
    generator_loss = []
    for epoch in range(epochs):
        batch = 1
        for X_batch in dataset:
            #Train discriminator
            batch_size = len(X_batch)
            noise = tf.random.normal(shape=[batch_size, coding_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]]*batch_size + [[1.]]*batch_size)
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X_fake_and_real, y1)
            discriminator_loss.append(d_loss)
            #Train generator
            noise = tf.random.normal(shape=[batch_size, coding_size])
            y2 = tf.constant([[1.]]*batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y2)
            generator_loss.append(g_loss)
            batch += 1
        print(f"Epoch: {epoch+1}, Discriminator loss: {discriminator_loss[-1]}, Generator loss: {generator_loss[-1]}")
    losses = {"d_loss": discriminator_loss, "g_loss": generator_loss}
    return losses