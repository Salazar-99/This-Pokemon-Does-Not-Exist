import tensorflow as tf
import math

PI = tf.constant(math.pi)

def vae_loss(model, input):
    z = model.encode(input)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=input)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def log_normal_pdf(sample, mean, log_var):
    log_2pi = tf.math.log(2. * PI)
    return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-log_var) + log_var + log_2pi),
      axis=1)
