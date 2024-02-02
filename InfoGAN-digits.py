#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import multiprocessing
import platform
import time


class infoGAN_digits():
    def __init__(self):
        self.generator = self._model_generator()
        self.discriminator = self._model_discriminator()
        self.Q = self._model_Q()

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator,
                                              Q=self.Q)

        self.checkpoint_path = "checkpoint/"
        self.checkpoint_file_prefix = "infoGAN"
        self.checkpint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_path, max_to_keep=10,
            checkpoint_name=self.checkpoint_file_prefix)
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        return

    def _model_generator(self):
        class Generator(tf.keras.Model):
            def __init__(self):
                super(Generator, self).__init__()

                self.d1 = tf.keras.layers.Dense(1024, use_bias=False)
                self.a1 = tf.keras.layers.ReLU()
                self.b1 = tf.keras.layers.BatchNormalization()
                self.d2 = tf.keras.layers.Dense(7 * 7 * 128, use_bias=False)
                self.a2 = tf.keras.layers.ReLU()
                self.b2 = tf.keras.layers.BatchNormalization()
                self.r2 = tf.keras.layers.Reshape([7, 7, 128])

                self.c3 = tf.keras.layers.Conv2DTranspose(
                    64, (4, 4), strides=(2, 2), padding="same")
                self.a3 = tf.keras.layers.ReLU()
                self.b3 = tf.keras.layers.BatchNormalization()

                self.c4 = tf.keras.layers.Conv2DTranspose(
                    1, (4, 4), strides=(2, 2), padding="same")

            def call(self, x, training=True):
                x = self.d1(x)
                x = self.b1(x, training=training)
                x = self.a1(x)

                x = self.d2(x)
                x = self.b2(x, training=training)
                x = self.a2(x)
                x = self.r2(x)

                x = self.c3(x)
                x = self.b3(x, training=training)
                x = self.a3(x)

                x = self.c4(x)

                x = tf.nn.tanh(x)

                return x

        g = Generator()
        return g

    def _model_discriminator(self):
        class Discriminator(tf.keras.Model):
            def __init__(self):
                super(Discriminator, self).__init__()

                self.c1 = tf.keras.layers.Conv2D(
                    64, (4, 4), strides=(2, 2), padding="same")
                self.a1 = tf.keras.layers.LeakyReLU()

                self.c2 = tf.keras.layers.Conv2D(
                    128, (4, 4), strides=(2, 2), padding="same")
                self.a2 = tf.keras.layers.LeakyReLU()
                self.b2 = tf.keras.layers.BatchNormalization()
                self.f2 = tf.keras.layers.Flatten()

                self.d3 = tf.keras.layers.Dense(1024)
                self.a3 = tf.keras.layers.LeakyReLU()
                self.b3 = tf.keras.layers.BatchNormalization()

                self.D = tf.keras.layers.Dense(1)

            def call(self, x, training=True):
                x = self.c1(x)
                x = self.a1(x)

                x = self.c2(x)
                x = self.b2(x, training=training)
                x = self.a2(x)
                x = self.f2(x)

                x = self.d3(x)
                x = self.b3(x, training=training)
                x = self.a3(x)

                mid = x

                D = self.D(x)

                return D, mid

        d = Discriminator()
        return d

    def _model_Q(self):
        class QNet(tf.keras.Model):
            def __init__(self):
                super(QNet, self).__init__()

                # base model part
                self.base_layer = [None for i in range(0, 3)]

                self.base_layer[0] = tf.keras.layers.Dense(128)
                self.base_layer[1] = tf.keras.layers.BatchNormalization()
                self.base_layer[2] = tf.keras.layers.LeakyReLU()

                # c_hat_layer
                self.c_hat_branch = [None for i in range(0, 1)]
                self.c_hat_branch[0] = tf.keras.layers.Dense(10)

                # z_hat_layer
                # self.z_hat_layer_mu=tf.keras.layers.Dense(2)
                # self.z_hat_layer_var=tf.keras.layers.Dense(2)

            def call(self, x, training=True):
                idx = 0
                x = self.base_layer[idx](x)
                idx += 1
                x = self.base_layer[idx](x, training=training)
                idx += 1
                x = self.base_layer[idx](x)
                del idx
                c_hat = self.c_hat_branch[0](x)
                return c_hat

        q = QNet()
        return q

    def generator_input(self, batch, digit_value=None):
        z = tfd.Uniform(low=-1.0, high=1.0).sample((batch, 64))
        if digit_value is None:
            c = tfd.Categorical(probs=tf.ones([10]) * 0.1).sample([batch, ])
        else:
            c = tf.ones([batch], dtype=tf.int32) * digit_value
        c = tf.one_hot(c, 10)
        return z, c

    def calc_info_loss(self, c, c_hat):
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)(c, c_hat)
        return loss

    def calc_generator_loss(self, fake_result):
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = loss_function(tf.ones_like(fake_result), fake_result)
        return loss

    def calc_discriminator_loss(self, real_result, fake_result):
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_function(tf.ones_like(real_result), real_result)
        fake_loss = loss_function(tf.zeros_like(fake_result), fake_result)
        loss = real_loss + fake_loss
        return loss

    def train_step(self, image, batch_size):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            z, c = self.generator_input(batch_size)
            generator_images = gan.generator(
                tf.concat([z, c], axis=-1), training=True)
            discriminator_fake, Q_input = self.discriminator(
                generator_images, training=True)
            discriminator_real, _ = self.discriminator(image, training=True)
            c_hat = self.Q(Q_input)
            info_loss = self.calc_info_loss(c, c_hat)
            generator_loss = self.calc_generator_loss(discriminator_fake)
            discriminator_loss = self.calc_discriminator_loss(
                discriminator_real, discriminator_fake)

            generator_infoGAN_loss = info_loss + generator_loss
            discriminator_infoGAN_loss = info_loss + discriminator_loss

        generator_gradient = generator_tape.gradient(generator_infoGAN_loss,
                                                     self.generator.trainable_variables + self.Q.trainable_variables)
        discriminator_gradient = discriminator_tape.gradient(discriminator_infoGAN_loss,
                                                             self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradient, self.generator.trainable_variables + self.Q.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradient, self.discriminator.trainable_variables))
        return info_loss, generator_loss, discriminator_loss

    def process_dataset(self):
        train, test = tf.keras.datasets.mnist.load_data()
        images, labels = train
        images = images.reshape([-1, 28, 28, 1]).astype("float32")
        images = (images - 255.0 / 2) / (255.0 / 2)
        return images

    def train(self, dataset, batch_size=200, epochs=400):
        batch_dataset = tf.data.Dataset.from_tensor_slices(
            dataset).shuffle(30000).batch(batch_size)
        batch_len = len(batch_dataset)
        train_cycle = 0
        d_loss_log = []
        g_loss_log = []
        REPORT_PERIOD_SEC = 30
        next_report_time = time.monotonic() + REPORT_PERIOD_SEC
        for epoch in range(0, epochs):
            for batch_cnt, batch_images in enumerate(batch_dataset):
                train_cycle += 1
                info_loss, generator_loss, discriminator_loss = self.train_step(
                    batch_images, batch_size)
                d_loss_log.append(discriminator_loss)
                g_loss_log.append(generator_loss)
                status = "[epoch:%d/%d batch:%d/%d]info_loss=%.5f,generator_loss=%.5f,discriminator_loss=%.5f" % (
                    epoch + 1, epochs, batch_cnt + 1, batch_len, info_loss, generator_loss, discriminator_loss)
                self._overwrite_print(status)
                now_monotonic_time = time.monotonic()
                if now_monotonic_time > next_report_time:
                    next_report_time += REPORT_PERIOD_SEC
                    dump_file_path = "train_images" + os.sep + \
                                     "infoGAN-step-%d.png" % (train_cycle)
                    self.debug_dump_model_image(d_loss_log, g_loss_log, dump_file_path)
                    self.save_model()

        return

    def _overwrite_print(self, s):
        try:
            white_out_str = ""
            for i in range(0, self._last_overwrite_print_length):
                white_out_str += " "
            print('\r' + white_out_str + '\r', end='')
        except AttributeError as e:
            pass
        print(s, end='\r')
        self._last_overwrite_print_length = len(s)
        return

    def run(self):
        dataset = self.process_dataset()
        self.train(dataset)
        return

    def save_model(self):
        self.checkpint_manager.save()
        return

    def load_model(self):
        self.checkpint_manager.restore_or_initialize()
        return

    def generate_image(self, batch, digit):
        z, c = self.generator_input(batch, digit)
        ginput = tf.concat([z, c], axis=-1)
        result = self.generator(ginput, training=False)
        result = np.array(result)
        result = result * (255 / 2) + (255 / 2)
        result = result.astype(int)
        return result

    def debug_peek_model_image(self):
        digits = [np.random.randint(0, 9) for i in range(25)]
        image_list = []
        for dig in digits:
            image_generate = self.generate_image(1, dig)
            image_generate = image_generate[0][:, :, 0]

            image_list.append(image_generate)

        image_w_cnt = 5
        image_h_cnt = 5
        plt.figure(figsize=(5, 5))
        for c in range(0, image_w_cnt):
            for r in range(0, image_h_cnt):
                index = c * image_h_cnt + r
                ax = plt.subplot(image_w_cnt, image_h_cnt, index + 1)
                plt.imshow(image_list[index])
                plt.axis("off")
        plt.show()
        return

    def debug_dump_model_image(self, d_loss_log, g_loss_log, fileName):
        digits = [int(i / 10) for i in range(100)]
        image_list = []
        for dig in digits:
            image_generate = self.generate_image(1, dig)
            image_generate = image_generate[0][:, :, 0]
            image_generate = 255 - np.array(image_generate)
            image_list.append(image_generate)

        image_w_cnt = 10
        image_h_cnt = 10

        def _plot(image_w_cnt, image_h_cnt, image_list, d_loss_log, g_loss_log, fileName):
            dir_name = os.path.dirname(fileName)
            dir_name = os.path.realpath(dir_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            fig, ax = plt.subplots(dpi=300, figsize=(16, 9))
            epoch = len(d_loss_log)
            _x = [i for i in range(0, epoch)]

            ax.plot(_x, d_loss_log, label="discriminator loss")
            ax.plot(_x, g_loss_log, label="generator loss")
            ax.set_xlabel("epoch/tick")
            ax.set_ylabel('loss')
            ax.set_title('D-G loss')
            ax.legend()
            ax.grid(True)

            fig.savefig("infoGAN_loss_plot.png")
            plt.close(fig)

            fig, ax = plt.subplots(image_w_cnt, image_h_cnt, figsize=(10, 10))
            for c in range(0, image_w_cnt):
                for r in range(0, image_h_cnt):
                    index = c * image_h_cnt + r
                    sub_plot = ax[c, r]
                    sub_plot.imshow(image_list[index],
                                    cmap='gray', vmin=0, vmax=255)
                    sub_plot.set_yticks([])
                    sub_plot.set_xticks([])

            fig.savefig(fileName, dpi=600)
            return

        if platform.system() == "Linux":
            if not hasattr(self, "imagePlotProces"):
                self.imagePlotProces = None
            if self.imagePlotProces is not None:
                self.imagePlotProces.join()
                self.imagePlotProces = None
            process = multiprocessing.Process(target=_plot, args=(
                image_w_cnt, image_h_cnt, image_list, d_loss_log, g_loss_log, fileName,))
            process.start()
            self.imagePlotProces = process
        elif platform.system() == "Windows":
            _plot(image_w_cnt, image_h_cnt, image_list, d_loss_log, g_loss_log, fileName)
        else:
            print("!!! should not be here !!!")
        return


if __name__ == "__main__":
    gan = infoGAN_digits()
    gan.load_model()
    dataset = gan.process_dataset()
    gan.train(dataset, batch_size=64, epochs=100)
    sys.exit(0)
