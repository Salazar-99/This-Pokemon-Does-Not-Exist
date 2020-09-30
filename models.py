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
            filters=3, kernel_size=3, strides=1, padding='same'))
    
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

#TODO: Refactor GAN into components so as to not subclass tf.keras.Model
class GAN(tf.keras.Model):
    def __init__(self, coding_size):
        super().__init__()
        self.coding_size = coding_size

        #Build generator
        self.generator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(15*15*32, input_shape=[coding_size]),
            tf.keras.layers.Reshape([15,15,32]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='tanh'),
        ])

        #Build discriminator
        self.discriminator = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation=self.lrelu(0.2), input_shape=[120,120,3]),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation=self.lrelu(0.2)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation=self.lrelu(0.2)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    #Components in here depend on components created in constructor
    def build(self, input_shape):
        #Build full model
        self.gan = tf.keras.models.Sequential([self.generator, self.discriminator])

    #Boilerplate for subclassing
    def call(self, inputs):
        pass

    #Utility for use in training  
    def get_components(self):
        return self.generator, self.discriminator
    
    #Convenience function for leak ReLU
    def lrelu(self, x):
        return tf.keras.layers.LeakyReLU(x)

def train_gan(gan, dataset, batch_size, coding_size, epochs):
    generator, discriminator = gan.get_components()
    #Set appropriate compilation configuration for training
    discriminator.compile(loss='binary_crossentropy', optimizer='rmsprop')
    discriminator.trainable = False
    gan.compile(loss='binary_crossentropy', optimizer='rmsprop')
    discriminator_loss = []
    generator_loss = []
    for epoch in range(epochs):
        batch = 1
        for X_batch in dataset:
            #Train discriminator
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
        print(f"Epoch: {epoch}, Batch: {batch}, Discriminator loss: {discriminator_loss[-1]}, Generator loss: {generator_loss[-1]}")
    losses = {"d_loss": discriminator_loss, "g_loss": generator_loss}
    return losses