import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from base.base_trainer import BaseTrain
import time

class CycleGANModelTrainer(BaseTrain):
    def __init__(self, model, trainA_data, trainB_data, testA_data, testB_data, config, tensorboard_log_dir, checkpoint_dir, sample_horse):
        super(CycleGANModelTrainer, self).__init__(model, trainA_data, trainB_data, testA_data, testB_data, config)
        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_dir = checkpoint_dir   
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.sample_horse = sample_horse

        ckpt = tf.train.Checkpoint(g_AB=self.model.g_AB,
                                g_BA=self.model.g_BA,
                                d_A=self.model.d_A,
                                d_B=self.model.d_B,
                                g_AB_optimizer=self.model.g_AB_optimizer,
                                g_BA_optimizer=self.model.g_BA_optimizer,
                                d_A_optimizer=self.model.d_A_optimizer,
                                d_B_optimizer=self.model.d_B_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    def generate_images(model, test_input):
        prediction = model(test_input)
            
        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    # TODO: save generated images to file folder
    # def sample_images(self, epoch, batch_i):
    #     os.makedirs('images/%s' % self.config['dataset_name'], exist_ok=True)
    #     r, c = 2, 3

    #     testA_datagen = DataGenerator(img_filenames=self.testA_data, batch_size=1, target_size=(self.config['img_height'], self.config['img_width']))
    #     testB_datagen = DataGenerator(img_filenames=self.testB_data, batch_size=1, target_size=(self.config['img_height'], self.config['img_width']))

    #     testA_generator = iter(testA_datagen)
    #     testB_generator = iter(testB_datagen)

    #     imgs_A = next(testA_generator)
    #     imgs_B = next(testB_generator)

    #     # Translate images to the other domain
    #     fake_B = self.model.g_AB.predict(imgs_A)
    #     fake_A = self.model.g_BA.predict(imgs_B)

    #     # Translate back to original domain
    #     reconstr_A = self.model.g_BA.predict(fake_B)
    #     reconstr_B = self.model.g_AB.predict(fake_A)

    #     gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

    #     # Rescale images 0 - 1
    #     gen_imgs = 0.5 * gen_imgs + 0.5

    #     # Plot images
    #     titles = ['Original', 'Translated', 'Reconstructed']
    #     fig, axs = plt.subplots(r, c)
    #     cnt = 0
    #     for i in range(r):
    #         for j in range(c):
    #             axs[i,j].imshow(gen_imgs[cnt])
    #             axs[i, j].set_title(titles[j])
    #             axs[i,j].axis('off')
    #             cnt += 1
    #     fig.savefig("images/%s/%d_%d.jpg" % (self.config['dataset_name'], epoch, batch_i))
    #     plt.close()

    #     imageio.imwrite("images/%s/%d_%d_a_transl.jpg" % (self.config['dataset_name'], epoch, batch_i), ((fake_B[0]+1)*127.5).astype(np.uint8))
    #     imageio.imwrite("images/%s/%d_%d_b_transl.jpg" % (self.config['dataset_name'], epoch, batch_i), ((fake_A[0]+1)*127.5).astype(np.uint8))
    #     imageio.imwrite("images/%s/%d_%d_a_recon.jpg" % (self.config['dataset_name'], epoch, batch_i), ((reconstr_A[0]+1)*127.5).astype(np.uint8))
    #     imageio.imwrite("images/%s/%d_%d_b_recon.jpg" % (self.config['dataset_name'], epoch, batch_i), ((reconstr_B[0]+1)*127.5).astype(np.uint8))

    @tf.function
    def train_step(self, real_A, real_B):
        # persistent is set to True because gen_tape and disc_tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(
            persistent=True) as disc_tape:
            
            fake_B = self.model.g_AB(real_A, training=True)
            cycled_A = self.model.g_BA(fake_B, training=True)

            fake_A = self.model.g_BA(real_B, training=True)
            cycled_B = self.model.g_AB(fake_A, training=True)

            # same_A and same_B are used for identity loss.
            same_A = self.model.g_BA(real_A, training=True)
            same_B = self.model.g_AB(real_B, training=True)

            disc_real_A = self.model.d_A(real_A, training=True)
            disc_real_B = self.model.d_B(real_B, training=True)

            disc_fake_A = self.model.d_A(fake_A, training=True)
            disc_fake_B = self.model.d_B(fake_B, training=True)

            # calculate the loss
            gen_AB_loss = self.model.generator_loss(disc_fake_B)
            gen_BA_loss = self.model.generator_loss(disc_fake_A)
            
            # Total generator loss = adversarial loss + cycle loss
            total_gen_AB_loss = gen_AB_loss + self.model.calc_cycle_loss(real_A, cycled_A) + self.model.identity_loss(real_B, same_B)
            total_gen_BA_loss = gen_BA_loss + self.model.calc_cycle_loss(real_B, cycled_B) + self.model.identity_loss(real_A, same_A)

            disc_A_loss = self.model.discriminator_loss(disc_real_A, disc_fake_A)
            disc_B_loss = self.model.discriminator_loss(disc_real_B, disc_fake_B)
        
        # Calculate the gradients for generator and discriminator
        generator_g_gradients = gen_tape.gradient(total_gen_AB_loss, 
                                                    self.model.g_AB.trainable_variables)
        generator_f_gradients = gen_tape.gradient(total_gen_BA_loss, 
                                                    self.model.g_BA.trainable_variables)
        
        self.discriminator_A_gradients = disc_tape.gradient(
            disc_A_loss, self.model.d_A.trainable_variables)
        discriminator_B_gradients = disc_tape.gradient(
            disc_B_loss, self.model.d_B.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.model.g_AB_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                    self.model.g_AB.trainable_variables))

        self.model.g_BA_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                    self.model.g_BA.trainable_variables))
        
        self.model.d_A_optimizer.apply_gradients(
            zip(self.discriminator_A_gradients,
            self.model.d_A.trainable_variables))
        
        self.model.d_B_optimizer.apply_gradients(
            zip(discriminator_B_gradients,
            self.model.d_B.trainable_variables))


    def train(self):
        epochs = self.config['nb_epoch']
        for epoch in range(epochs):
            start = time.time()

            n = 0
            for image_x, image_y in tf.data.Dataset.zip((self.trainA_data, self.trainB_data)):
                self.train_step(image_x, image_y)
                if n % 10 == 0:
                    print ('.', end='')
                n+=1

            # Using a consistent image (sample_horse) so that the progress of the model
            # is clearly visible.
            generate_images(self.model.g_AB, self.sample_horse)

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                    ckpt_save_path))

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                time.time()-start))

                    



        