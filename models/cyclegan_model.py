import tensorflow as tf
from base.base_model import BaseModel
from tensorflow_examples.models.pix2pix import pix2pix

class CycleGANModel(BaseModel):
    def __init__(self, config, weights_path, is_train=True):
        super(CycleGANModel, self).__init__(config)
        self.CHANNELS = 3
        self.IMG_HEIGHT = config['img_height']
        self.IMG_WIDTH = config['img_width']
        self.IMG_SHAPE = (self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNELS)
        self.WEIGHTS_PATH = weights_path
        self.build_model()
        
    def build_model(self):
        self.g_AB = pix2pix.unet_generator(self.CHANNELS, norm_type='instancenorm')
        self.g_BA = pix2pix.unet_generator(self.CHANNELS, norm_type='instancenorm')

        self.d_A = pix2pix.discriminator(norm_type='instancenorm', target=False)
        self.d_B = pix2pix.discriminator(norm_type='instancenorm', target=False)

        self.g_AB_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.g_BA_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.d_A_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.d_B_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

 


