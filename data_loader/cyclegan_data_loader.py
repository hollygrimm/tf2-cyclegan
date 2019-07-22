
import tensorflow as tf
import tensorflow_datasets as tfds
from base.base_data_loader import BaseDataLoader


class CycleGANDataLoader(BaseDataLoader):

    def __init__(self, config):
        """Initialize dataset
        Args:
            config : dictionary
        """
        super(CycleGANDataLoader, self).__init__(config)
        self.BUFFER_SIZE = config['buffer_size']
        self.BATCH_SIZE = config['batch_size']
        self.IMG_HEIGHT = config['img_height']
        self.IMG_WIDTH = config['img_width']

        # delegate the setting of num_parallel_calls to tf.data runtime
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                                      with_info=True, as_supervised=True)

        self.train_a, self.train_b = dataset['trainA'], dataset['trainB']
        self.test_a, self.test_b = dataset['testA'], dataset['testB']

        # transform the data, cache, shuffle, and batch
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
        """Crop the image randomly to a given size
        Args:
            image : tensor
        Returns:
            cropped image (tensor)
        """
        cropped_image = tf.image.random_crop(
            image, size=[self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        return cropped_image

    def normalize(self, image):
        """Normalize the image data to [-1, 1]
        Args:
            image : tensor
        Returns:
            normalized image (tensor)
        """
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image

    def random_jitter(self, image):
        """Increase height and width by 30 pixels, random crop back to orig size, and randomly flip left to right
        Args:
            image : tensor
        Returns:
            image w/random jitter (tensor)
        """
        SIZE_INCREASE = 30
        image = tf.image.resize(image, [self.IMG_HEIGHT + SIZE_INCREASE, self.IMG_WIDTH + SIZE_INCREASE],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # randomly crop back to 256 x 256 x 3
        image = self.random_crop(image)
        image = tf.image.random_flip_left_right(image)
        return image

    def preprocess_image_train(self, image, label):
        """Apply random jitter, then normalize image
        Args:
            image : tensor
        Returns:
            image w/random jitter and normalization (tensor)
        """
        image = self.random_jitter(image)
        image = self.normalize(image)
        return image

    def preprocess_image_test(self, image, label):
        """Normalize image
        Args:
            image : tensor
        Returns:
            normalized image (tensor)
        """
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
