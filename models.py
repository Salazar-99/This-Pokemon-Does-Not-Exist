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
