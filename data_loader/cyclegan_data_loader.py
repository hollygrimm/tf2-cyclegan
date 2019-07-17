
import tensorflow as tf
import tensorflow_datasets as tfds
from base.base_data_loader import BaseDataLoader

class CycleGANDataLoader(BaseDataLoader):


    def __init__(self, config):
        super(CycleGANDataLoader, self).__init__(config)

        self.BUFFER_SIZE = config['buffer_size']
        self.BATCH_SIZE = config['batch_size']
        self.IMG_WIDTH = config['img_size']
        self.IMG_HEIGHT = config['img_size']
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE        

        dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                                    with_info=True, as_supervised=True)

        self.train_a, self.train_b = dataset['trainA'], dataset['trainB']
        self.test_a, self.test_b = dataset['testA'], dataset['testB']
        

        self.train_a = self.train_a.map(
            self.preprocess_image_train, num_parallel_calls=self.AUTOTUNE).cache().shuffle(
            self.BUFFER_SIZE).batch(self.BATCH_SIZE)

        self.train_b = self.train_b.map(
            self.preprocess_image_train, num_parallel_calls=self.AUTOTUNE).cache().shuffle(
            self.BUFFER_SIZE).batch(self.BATCH_SIZE)

        self.test_a = self.test_a.map(
            self.preprocess_image_test, num_parallel_calls=self.AUTOTUNE).cache().shuffle(
            self.BUFFER_SIZE).batch(self.BATCH_SIZE)

        self.test_b = self.test_b.map(
            self.preprocess_image_test, num_parallel_calls=self.AUTOTUNE).cache().shuffle(
            self.BUFFER_SIZE).batch(self.BATCH_SIZE)


    def random_crop(self, image):
        cropped_image = tf.image.random_crop(
        image, size=[self.IMG_HEIGHT, self.IMG_WIDTH, 3])

        return cropped_image


    # normalizing the images to [-1, 1]
    def normalize(self, image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image


    def random_jitter(self, image):
        # resizing to 286 x 286 x 3
        image = tf.image.resize(image, [286, 286],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        image = self.random_crop(image)

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image

    def preprocess_image_train(self, image, label):
        image = self.random_jitter(image)
        image = self.normalize(image)
        return image

    def preprocess_image_test(self, image, label):
        image = self.normalize(image)
        return image

    def get_train_a_data(self):
        return self.train_a

    def get_train_b_data(self):
        return self.train_b   

    def get_test_a_data(self):
        return self.test_a

    def get_test_b_data(self):
        return self.test_b     








    
