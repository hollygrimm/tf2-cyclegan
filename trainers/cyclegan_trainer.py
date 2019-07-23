import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from IPython.display import clear_output
import imageio
import tensorflow as tf
from base.base_trainer import BaseTrain
import time

class CycleGANModelTrainer(BaseTrain):
    def __init__(self, model, trainA_data, trainB_data, testA_data, testB_data, config, tensorboard_log_dir, checkpoint_dir, image_dir, viz_notebook=False):
        super(CycleGANModelTrainer, self).__init__(model, trainA_data, trainB_data, testA_data, testB_data, config)
        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_dir = checkpoint_dir
        self.image_dir = image_dir
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.viz_notebook = viz_notebook

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

        self.train_summary_writer = tf.summary.create_file_writer(tensorboard_log_dir)

    def generate_images(self, model, input_image):
        prediction = model(input_image)
            
        plt.figure(figsize=(12, 12))

        display_list = [input_image[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    # FIXME: epoch vs id
    def save_generated_images(self, model, input_image, input_type, epoch, img_id):
        prediction = model(input_image)

        if epoch != None:
            imageio.imwrite("%s%d_%s_transl.jpg" % (self.image_dir, epoch, input_type), ((prediction[0]+1)*127.5))
        else:
            imageio.imwrite("%s%s_transl_%d.jpg" % (self.image_dir, input_type, img_id), ((prediction[0]+1)*127.5))


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

            gen_A_cycle_loss = self.model.calc_cycle_loss(real_A, cycled_A)
            gen_B_cycle_loss = self.model.calc_cycle_loss(real_B, cycled_B)

            gen_A_identity_loss = self.model.identity_loss(real_A, same_A)
            gen_B_identity_loss = self.model.identity_loss(real_B, same_B)
            
            # Total generator loss = adversarial loss + cycle loss
            total_gen_AB_loss = gen_AB_loss + gen_A_cycle_loss + gen_B_identity_loss
            total_gen_BA_loss = gen_BA_loss + gen_B_cycle_loss + gen_A_identity_loss

            disc_A_loss = self.model.discriminator_loss(disc_real_A, disc_fake_A)
            disc_B_loss = self.model.discriminator_loss(disc_real_B, disc_fake_B)

        # log losses for tensorboard
        gen_AB_losses = {'gen_AB_loss': gen_AB_loss,
                            'gen_A_cycle_loss': gen_A_cycle_loss,
                            'gen_B_identity_loss': gen_B_identity_loss}

        gen_BA_losses = {'gen_BA_loss': gen_BA_loss,
                            'gen_B_cycle_loss': gen_B_cycle_loss,
                            'gen_A_identity_loss': gen_A_identity_loss}                            

        for name, value in gen_AB_losses.items():
            tf.summary.scalar(name, value, step=self.model.g_AB_optimizer.iterations)

        for name, value in gen_BA_losses.items():
            tf.summary.scalar(name, value, step=self.model.g_BA_optimizer.iterations)            

        tf.summary.scalar('d_A_loss', disc_A_loss, step=self.model.d_A_optimizer.iterations)

        tf.summary.scalar('d_B_loss', disc_B_loss, step=self.model.d_B_optimizer.iterations)        
        
        # Calculate the gradients for generator and discriminator
        gen_AB_gradients = gen_tape.gradient(total_gen_AB_loss, 
                                                    self.model.g_AB.trainable_variables)
        gen_BA_gradients = gen_tape.gradient(total_gen_BA_loss, 
                                                    self.model.g_BA.trainable_variables)
        
        self.discriminator_A_gradients = disc_tape.gradient(
            disc_A_loss, self.model.d_A.trainable_variables)
        discriminator_B_gradients = disc_tape.gradient(
            disc_B_loss, self.model.d_B.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.model.g_AB_optimizer.apply_gradients(zip(gen_AB_gradients, 
                                                    self.model.g_AB.trainable_variables))

        self.model.g_BA_optimizer.apply_gradients(zip(gen_BA_gradients, 
                                                    self.model.g_BA.trainable_variables))
        
        self.model.d_A_optimizer.apply_gradients(
            zip(self.discriminator_A_gradients,
            self.model.d_A.trainable_variables))
        
        self.model.d_B_optimizer.apply_gradients(
            zip(discriminator_B_gradients,
            self.model.d_B.trainable_variables))


    def train(self):
        # Predict using a consistent image (sample from a and b) so that the progress of the model
        # is clearly visible.
        sample_a = next(iter(self.trainA_data))
        sample_b = next(iter(self.trainB_data))

        with self.train_summary_writer.as_default():
            epochs = self.config['nb_epoch']
            for epoch in range(epochs):
                start = time.time()

                n = 0
                for image_x, image_y in tf.data.Dataset.zip((self.trainA_data, self.trainB_data)):
                    self.train_step(image_x, image_y)
                    if n % 10 == 0:
                        print ('.', end='')
                    n += 1

                if self.viz_notebook:
                    clear_output(wait=True)
                    self.generate_images(self.model.g_AB, sample_a)
                    self.generate_images(self.model.g_BA, sample_b)
                else:
                    self.save_generated_images(self.model.g_AB, sample_a, "a", epoch, None)
                    self.save_generated_images(self.model.g_BA, sample_b, "b", epoch, None)

                if (epoch + 1) % 5 == 0:
                    ckpt_save_path = self.ckpt_manager.save()
                    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                        ckpt_save_path))

                print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                    time.time()-start))

                    
    def predict(self):
        if self.viz_notebook:
            for testA in self.testA_data.take(5):
                self.generate_images(self.model.g_AB, testA)
            for testB in self.testB_data.take(5):
                self.generate_images(self.model.g_BA, testB)
        else:
            for i, testA in enumerate(self.testA_data.take(5)):
                self.save_generated_images(self.model.g_AB, testA, "a", None, i)
            for i, testB in enumerate(self.testB_data.take(5)):
                self.save_generated_images(self.model.g_BA, testB, "b", None, i)



        
