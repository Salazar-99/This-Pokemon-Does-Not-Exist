import tensorflow as tf
import tensorflow_probability as tfp 

class VAE(tf.keras.Model):
    def __init__(self, conv_layers, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.conv_layers = conv_layers
        
        #Build encoder
        self.encoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(120,120,3))])
        for i in range(conv_layers):
            filters = get_filters(i)
            self.encoder.add(tf.keras.layers.Conv2D(
                filters=filters, kernel_size=3, strides=(2,2), padding='same', acitvation='relu'))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(latent_dims))
        self.encoder.add(Sampler())
        
        #Build decoder
        self.decoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dims,))])
        self.decoder.add(tf.keras.layers.Dense(units=7*7*32, activation='relu')
        self.decoder.add(tf.keras.layers.Reshape(target_shape=(7, 7, 32)))
        for i in range(conv_layers-1):
            filters = get_filters(conv_layers-i)
            self.encoder.add(tf.keras.layers.Conv2DTranspose(
                filters=filters, kernel_size=3, strides=2, padding='same', acitvation='relu'))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=1, padding='same'))

    #Full forward pass
    def call(self, inputs):
        encoded_image = self.encoder(inputs)
        decoded_image = self.decoder(encoded_image)
        return decoded_image

    #Utility for encoding data
    def encode(self, x):
        return self.encoder(x)

    #Utility for generating new data
    def decode(self, z):
        return self.decoder(z)
    
    #Computing number of filters as a function of depth
    def get_filters(i):
        return 2**(i+4)

    #Reparameterization trick implemented as a layer
    class Sampler(tf.keras.layers.Layer):
        def call(self, inputs):
            mean, log_var = inputs
            return tf.random.normal(tf.shape(log_var)) * tf.math.exp(log_var/2) + mean


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
            tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation=lrelu(0.2), input_shape=[120,120,3]),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation=lrelu(0.2)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation=lrelu(0.2)),
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