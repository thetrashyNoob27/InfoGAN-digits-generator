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
import inspect


class infoGAN_digits():
    def __init__(self, noise_dim, continuous_dim, categories_dim, mid_feautures):
        self.noise_dim = noise_dim
        self.continuous_dim = continuous_dim
        self.categories_dim = categories_dim
        self.mid_feautures = mid_feautures
        self.generator = self._model_generator(noise_dim, continuous_dim, categories_dim)
        self.discriminator_base = self._model_discriminator_base(mid_feautures)
        self.discriminator = self._model_discriminator(mid_feautures)
        self.quaility_control = self._model_quality_control(mid_feautures, continuous_dim, categories_dim)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator_base=self.discriminator_base,
                                              discriminator=self.discriminator,
                                              quaility_control=self.quaility_control)

        self.checkpoint_path = "checkpoint/"
        self.checkpoint_file_prefix = "infoGAN"
        self.checkpint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_path, max_to_keep=10,
            checkpoint_name=self.checkpoint_file_prefix)
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        return

    def _generator_internals(self, x):
        _layers = []
        _layers.append(tf.keras.layers.Dense(1024, use_bias=False))
        _layers.append(tf.keras.layers.ReLU())
        _layers.append(tf.keras.layers.BatchNormalization())

        _layers.append(tf.keras.layers.Dense(7 * 7 * 128, use_bias=False))
        _layers.append(tf.keras.layers.ReLU())
        _layers.append(tf.keras.layers.BatchNormalization())

        _layers.append(tf.keras.layers.Reshape([7, 7, 128]))

        _layers.append(tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same"))
        _layers.append(tf.keras.layers.ReLU())
        _layers.append(tf.keras.layers.BatchNormalization())

        _layers.append(tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding="same"))

        _layers.append(tf.keras.layers.Activation('tanh'))

        for l in _layers:
            x = l(x)
        return x

    def _model_generator(self, noise_dim, continuous_dim, categories_dim):
        noise = tf.keras.layers.Input(shape=(noise_dim,))
        continuous = tf.keras.layers.Input(shape=(continuous_dim,))
        category = tf.keras.layers.Input(shape=(categories_dim,))
        input_layer = tf.keras.layers.Concatenate()([noise, continuous, category])
        gen = self._generator_internals(input_layer)

        gen_model = tf.keras.Model(inputs=[noise, continuous, category], outputs=gen)
        return gen_model

    def _discriminator_base_internals(self, x):
        common_layers = []
        _layer_prefix = "discriminator_base_internals"

        common_layers.append(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", name=_layer_prefix + str(inspect.currentframe().f_lineno) + "_"))
        common_layers.append(tf.keras.layers.BatchNormalization())
        common_layers.append(tf.keras.layers.LeakyReLU())

        common_layers.append(tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same"))
        common_layers.append(tf.keras.layers.BatchNormalization())
        common_layers.append(tf.keras.layers.LeakyReLU())

        common_layers.append(tf.keras.layers.Flatten())

        common_layers.append(tf.keras.layers.Dense(1024))
        common_layers.append(tf.keras.layers.BatchNormalization())
        common_layers.append(tf.keras.layers.LeakyReLU())

        for l in common_layers:
            x = l(x)
        return x

    def _model_discriminator_base(self, mid_feautures):
        _layer_prefix = "model_discriminator_base"
        input_shape = self.generator.layers[-1].output_shape[1:]
        generator_output = tf.keras.layers.Input(shape=input_shape, name=_layer_prefix + str(inspect.currentframe().f_lineno) + "_")
        mid = self._discriminator_base_internals(generator_output)
        mid = tf.keras.layers.Dense(mid_feautures)(mid)

        discriminator_base_model = tf.keras.Model(inputs=generator_output, outputs=mid)

        return discriminator_base_model

    def _model_discriminator(self, mid_feautures):
        base_output = tf.keras.layers.Input(shape=(mid_feautures,))
        discrim = tf.keras.layers.Dense(1)(base_output)
        discrim = tf.keras.activations.sigmoid(discrim)
        discriminator_model = tf.keras.Model(inputs=base_output, outputs=discrim)
        return discriminator_model

    def _model_quality_control_internals(self, x):
        _layer_prefix = "model_quality_control_internals"
        qc_layers = []
        qc_layers.append(tf.keras.layers.Dense(128, name=_layer_prefix + str(inspect.currentframe().f_lineno) + "_"))
        qc_layers.append(tf.keras.layers.BatchNormalization())
        qc_layers.append(tf.keras.layers.LeakyReLU())
        for l in qc_layers:
            x = l(x)
        return x

    def _model_quality_control(self, mid_feautures, continuous, categorys):
        base_output = tf.keras.layers.Input(shape=(mid_feautures,))
        x = self._model_quality_control_internals(base_output)

        continue_ = tf.keras.layers.Dense(continuous, activation='linear', name="debug_" + str(inspect.currentframe().f_lineno))(x)
        classify = tf.keras.layers.Dense(categorys, activation='softmax', name="debug_" + str(inspect.currentframe().f_lineno))(x)

        qc_model = tf.keras.Model(inputs=base_output, outputs=[continue_, classify])
        return qc_model

    def generator_input(self, batch, digit_value=None):
        noise = np.random.uniform(-1, 1, (batch, self.noise_dim))
        continuous = np.random.uniform(-1, 1, (batch, self.continuous_dim))
        if digit_value is None:
            sampled_categories = np.random.randint(0, self.categories_dim, (batch, 1))
        else:
            sampled_categories=np.full((batch,1),digit_value)
        category = tf.keras.utils.to_categorical(sampled_categories, self.categories_dim)
        return noise, continuous, category

    def calc_info_loss(self, c, c_hat, a, a_hat):
        loss = tf.keras.losses.CategoricalCrossentropy()(c, c_hat) + tf.keras.losses.mse(a, a_hat)
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

    def discrimate(self, data):
        discriminator_intermident = self.discriminator_base(data)
        discrimination = self.discriminator(discriminator_intermident, training=True)
        analog, classify = self.quaility_control(discriminator_intermident, training=True)
        return discrimination, analog, classify

    def train_step(self, image, batch_size):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            z, ctn, cat = self.generator_input(batch_size)
            generator_images = gan.generator([z, ctn, cat], training=True)

            dis_fake, qc_analog, qc_classify = self.discrimate(generator_images)
            dis_real, _, _ = self.discrimate(image)

            info_loss = self.calc_info_loss(cat, qc_classify, ctn, qc_analog)
            generator_loss = self.calc_generator_loss(dis_fake)
            discriminator_loss = self.calc_discriminator_loss(dis_real, dis_fake)

            generator_infoGAN_loss = info_loss + generator_loss
            discriminator_infoGAN_loss = info_loss + discriminator_loss

        generator_gradient = generator_tape.gradient(generator_infoGAN_loss, self.generator.trainable_variables + self.discriminator_base.trainable_variables + self.quaility_control.trainable_variables)
        discriminator_gradient = discriminator_tape.gradient(discriminator_infoGAN_loss, self.discriminator_base.trainable_variables + self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradient, self.generator.trainable_variables + self.discriminator_base.trainable_variables + self.quaility_control.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator_base.trainable_variables + self.discriminator.trainable_variables))
        return np.mean(info_loss), np.mean(generator_loss), np.mean(discriminator_loss)

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
                info_loss, generator_loss, discriminator_loss = self.train_step(batch_images, batch_size)
                d_loss_log.append(discriminator_loss)
                g_loss_log.append(generator_loss)
                status = "[epoch:%d/%d batch:%d/%d]info_loss=%.5f,generator_loss=%.5f,discriminator_loss=%.5f" % (epoch + 1, epochs, batch_cnt + 1, batch_len, info_loss, generator_loss, discriminator_loss)
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
        z, a, c = self.generator_input(batch, digit)
        result = self.generator([z, a, c], training=False)
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
    if platform.system() == "Linux":
        os.nice(19)
    gan = infoGAN_digits(64, 1, 10, 1000)
    gan.load_model()
    dataset = gan.process_dataset()
    gan.train(dataset, batch_size=64, epochs=100)
    sys.exit(0)
