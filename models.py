import tensorflow as tf
from losses import vae_loss

class VAE(tf.keras.Model):
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
        for i in range(conv_layers-1):
            filters = self.get_filters(conv_layers-1-i)
            self.decoder.add(tf.keras.layers.Conv2DTranspose(
                filters=filters, kernel_size=(3,3), strides=2, padding='same', activation='relu'))
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
    losses = []
    optimizer = tf.keras.optimizers.Adam(lr)
    for epoch in range(epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                loss = vae_loss(vae, batch)
                losses.append(loss)
            gradients = tape.gradient(loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        print(f"Epoch: {epoch}, Loss: {losses[-1]}")
    return losses

class GAN(tf.keras.Model):
    def __init__(self, coding_size):
        super().__init__()
        self.coding_size = coding_size

        #Build generator
        self.generator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(6*6*256, input_shape=[coding_size]),
            tf.keras.layers.Reshape([15,15,256]),
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

        #Full model
        self.gan = tf.keras.models.Sequential([generator, discriminator])

        #Set appropriate compilation configuration for training
        self.discriminator.compile(loss='binary_crossentropy', optimizer='rmsprop')
        self.discriminator.trainable = False
        self.gan.compile(loss='binary_crossentropy', optimizer='rmsprop')

    #Utility for use in training  
    def get_components(self):
        return self.generator, self.disciminator
    
    #Convenience function for leak ReLU
    def lrelu(self, x):
        return tf.keras.layers.LeakyReLU(x)

def train_gan(gan, dataset, batch_size, coding_size, epochs):
    generator, discriminator = gan.get_components()
    for epoch in range(epochs):
        for X_batch in dataset:
            #Train discriminator
            noise = tf.random.normal(shape=[batch_shape, coding_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]]*batch_size + [[1.]]*batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            #Train generator
            noise = tf.random.normal(shape=[batch_size, coding_size])
            y2 = tf.constant([[1.]]*batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)