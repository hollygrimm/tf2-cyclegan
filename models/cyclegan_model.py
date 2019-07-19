import tensorflow as tf
from base.base_model import BaseModel
from tensorflow_examples.models.pix2pix import pix2pix

class CycleGANModel(BaseModel):
    def __init__(self, config, is_train=True):
        super(CycleGANModel, self).__init__(config)
        self.CHANNELS = 3
        self.IMG_HEIGHT = config['img_height']
        self.IMG_WIDTH = config['img_width']
        self.IMG_SHAPE = (self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNELS)
        self.LAMBDA = 10
        self.build_model()

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss
        
    def build_model(self):
        self.g_AB = pix2pix.unet_generator(self.CHANNELS, norm_type='instancenorm')
        self.g_BA = pix2pix.unet_generator(self.CHANNELS, norm_type='instancenorm')

        self.d_A = pix2pix.discriminator(norm_type='instancenorm', target=False)
        self.d_B = pix2pix.discriminator(norm_type='instancenorm', target=False)

        self.g_AB_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.g_BA_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.d_A_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.d_B_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


 


