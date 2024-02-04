#!/usr/bin/env python3
# example of training an infogan on mnist
import multiprocessing
import os
import platform
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(16)
tf.random.set_seed(16)


def build_generator(latent_dim, num_continuous, num_categories):
    noise = tf.keras.layers.Input(shape=(latent_dim,))
    continuous_input = tf.keras.layers.Input(shape=(num_continuous,))
    category_input = tf.keras.layers.Input(shape=(num_categories,))

    x = tf.keras.layers.Concatenate()(
        [noise, continuous_input, category_input])
    gen = x

    gen = tf.keras.layers.Dense(1024)(gen)
    gen = tf.keras.layers.Activation('relu')(gen)
    gen = tf.keras.layers.BatchNormalization()(gen)

    n_nodes = 512 * 7 * 7
    gen = tf.keras.layers.Dense(n_nodes)(gen)
    gen = tf.keras.layers.Activation('relu')(gen)
    gen = tf.keras.layers.BatchNormalization()(gen)
    gen = tf.keras.layers.Reshape((7, 7, 512))(gen)
    gen = tf.keras.layers.Dropout(0.05)(gen)
    # normal
    gen = tf.keras.layers.Conv2D(128, (4, 4), padding='same')(gen)
    gen = tf.keras.layers.Activation('relu')(gen)
    gen = tf.keras.layers.BatchNormalization()(gen)
    gen = tf.keras.layers.Dropout(0.05)(gen)
    # upsample to 14x14
    gen = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = tf.keras.layers.Activation('relu')(gen)
    gen = tf.keras.layers.BatchNormalization()(gen)
    gen = tf.keras.layers.Dropout(0.05)(gen)
    # upsample to 28x28
    gen = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same')(gen)
    # tanh output
    generated_image = tf.keras.layers.Activation('tanh', name="generator_output")(gen)

    return tf.keras.Model(inputs=[noise, continuous_input, category_input], outputs=generated_image)


def build_discriminator_base(datashape, intermident_units):
    image_input = tf.keras.layers.Input(shape=datashape)
    # downsample to 14x14
    d = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(image_input)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.1)(d)
    # downsample to 7x7
    d = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.1)(d)
    # normal

    # flatten feature maps
    d = tf.keras.layers.Flatten()(d)
    d = tf.keras.layers.Dense(intermident_units)(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.1)(d)

    intermident = d
    return tf.keras.Model(inputs=image_input, outputs=intermident)


def build_discriminator(intermident_units):
    mid = tf.keras.layers.Input(shape=(intermident_units,))
    validity = tf.keras.layers.Dense(1, name="real_fake_discrimination")(mid)
    discrimator_model = tf.keras.Model(inputs=mid, outputs=validity)
    return discrimator_model


def build_quality_control(intermident_units, num_continuous, num_categories):
    mid = tf.keras.layers.Input(shape=(intermident_units,))
    x = mid
    x = tf.keras.layers.Dense(23)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    continuous_output = tf.keras.layers.Dense(
        num_continuous, name="Q_linear")(x)
    category_output = tf.keras.layers.Dense(
        num_categories, name="Q_classify")(x)

    quility_control_model = tf.keras.Model(inputs=mid, outputs=[continuous_output, category_output])
    return quility_control_model


def remove_old_image():
    "infoGAN - %d.png"
    test_image_pattern = re.compile(r"infoGAN-(\d+)\.png")
    loss_image_pattern = re.compile(r"infoGAN_loss_plot_(\d+)\.png")
    rmlist = []
    for root, dirs, files in os.walk("."):
        for f in files:
            rm_mark = False
            if test_image_pattern.match(f):
                rm_mark = True
            elif loss_image_pattern.match(f):
                rm_mark = True

            if rm_mark:
                fpath = os.path.join(root, f)
                rmlist.append(fpath)

    # Remove each matching file
    for file_path in rmlist:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    return


def wassertein_loss(ytrue, yhat):
    loss = tf.reduce_mean(ytrue * yhat)
    return loss


if __name__ == "__main__":
    remove_old_image()
    if platform.system() == "Linux":
        os.nice(19)
    # Dimensions
    latent_dim = 62
    num_continuous = 1
    num_categories = 10
    num_mid_features = 1000

    # Build and compile the generator, discriminator, and InfoGAN models
    try:
        generator = tf.keras.models.load_model("infoGAN-model-G.tf")
        discriminator = tf.keras.models.load_model("infoGAN-model-D.tf")
        quility_control = tf.keras.models.load_model("infoGAN-model-Q.tf")
        print("load model success")
    except OSError as e:
        generator = build_generator(latent_dim, num_continuous, num_categories)

        discriminator_base = build_discriminator_base((28, 28, 1), num_mid_features)
        discriminator = build_discriminator(num_mid_features)
        quility_control = build_quality_control(num_mid_features, num_continuous, num_categories)

    # prepare dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    x = np.expand_dims(x, axis=-1)
    del x_train, y_train, x_test, y_test
    x_train = (x - 127.5) / 127.5
    x_train = x_train.astype("float32")

    # Build and compile the generator, discriminator, and InfoGAN models
    # (As per the previous example code)

    # Training parameters
    epochs = 10000
    batch_size = 64

    # Training loop
    REPORT_PERIOD_SEC = 60
    next_report_time = time.monotonic() + 30
    d_loss_log = []
    g_loss_log = []
    d_score_log = []
    g_score_log = []
    trainCnt = 0
    discriminator_optmizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    generator_optmizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    for epoch in range(epochs):
        indices = np.arange(0, x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y = y[indices]
        batchs = np.array_split(x_train, np.ceil(x_train.shape[0] / batch_size))
        for b in batchs:
            trainCnt += 1

            real_images = b
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            sampled_categories = np.random.randint(0, num_categories, batch_size)
            sampled_categories_one_hot = tf.keras.utils.to_categorical(sampled_categories, num_categories)
            del sampled_categories
            sampled_continuous = np.random.uniform(-1, 1, (batch_size, num_continuous))

            with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                generated_images = generator([noise, sampled_continuous, sampled_categories_one_hot], training=True)

                mid_fake = discriminator_base(generated_images, training=True)
                discriminator_fake = discriminator(mid_fake, training=True)
                mid_real = discriminator_base(real_images, training=True)
                discriminator_real = discriminator(mid_real, training=True)
                quility_control_continue, quility_control_classify = quility_control(mid_fake)

                valid = np.ones((batch_size, 1))
                fake = np.zeros((batch_size, 1))

                # loss
                quility_loss = tf.reduce_mean(tf.keras.losses.mse(sampled_continuous, quility_control_continue))
                quility_loss += tf.keras.losses.CategoricalCrossentropy(from_logits=True)(sampled_categories_one_hot, quility_control_classify)
                generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(valid, discriminator_fake)
                discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(fake, discriminator_fake) + tf.keras.losses.BinaryCrossentropy(from_logits=True)(valid, discriminator_real)

                info_g_loss = quility_loss + generator_loss
                info_d_loss = quility_loss + discriminator_loss

                generator_gradient = generator_tape.gradient(info_g_loss, generator.trainable_variables + quility_control.trainable_variables)
                discriminator_gradient = discriminator_tape.gradient(info_d_loss, discriminator.trainable_variables + discriminator_base.trainable_variables)

                discriminator_optmizer.apply_gradients(zip(generator_gradient, generator.trainable_variables + quility_control.trainable_variables))
                generator_optmizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables + discriminator_base.trainable_variables))

                generator_score = np.mean(discriminator_fake)
                discriminator_score = np.mean([np.mean(discriminator_real), 1 - generator_score])
                g_score_log.append(generator_score)
                d_score_log.append(discriminator_score)

                d_loss_log.append(np.mean(info_d_loss))
                g_loss_log.append(np.mean(info_g_loss))

                # Print progress
                plot_process = None
                now_monotonic_time = time.monotonic()
                if now_monotonic_time > next_report_time:
                    next_report_time += REPORT_PERIOD_SEC
                    print("[%d] D_loss%10.4f: G_loss:%10.4f" % (trainCnt, d_loss_log[-1], g_loss_log[-1]))

                    # save model
                    generator.save("infoGAN-model-G.tf", save_format="tf")
                    discriminator.save("infoGAN-model-D.tf", save_format="tf")
                    quility_control.save("infoGAN-model-Q.tf", save_format="tf")

                    # plot image process
                    noise = np.random.normal(0, 1, (100, latent_dim))
                    cat_array = np.zeros((100, num_categories))
                    for i in range(0, 100):
                        idx = i // num_categories
                    cat_array[i, idx] = 1
                    sampled_categories = cat_array
                    del cat_array

                    sampled_continuous = np.random.uniform(-1, 1, (100, num_continuous))

                    generated_images = generator.predict([noise, sampled_continuous, sampled_categories], verbose=0)


                    # print(generated_images.shape)#(100, 28, 28, 1)

                    def _plot(generated, epoch, d_loss_log, g_loss_log):
                        fig, ax = plt.subplots(2, 1, dpi=300, figsize=(16, 9))
                        _x = [i for i in range(0, epoch)]
                        _plot_idx = 0
                        ax[_plot_idx].plot(_x, d_loss_log, label="discriminator loss")
                        ax[_plot_idx].plot(_x, g_loss_log, label="generator loss")
                        ax[_plot_idx].set_xlabel("epoch/tick")
                        ax[_plot_idx].set_ylabel('loss')
                        ax[_plot_idx].set_title('D-G loss')
                        ax[_plot_idx].legend()
                        ax[_plot_idx].grid(True)

                        _plot_idx = 1
                        ax[_plot_idx].plot(_x, d_score_log, label="discriminator score")
                        ax[_plot_idx].plot(_x, g_score_log, label="generator score")
                        ax[_plot_idx].set_ylim(-0.1, 1.1)
                        ax[_plot_idx].set_xlabel("epoch/tick")
                        ax[_plot_idx].set_ylabel('score')
                        ax[_plot_idx].set_title('D-G score')
                        ax[_plot_idx].legend()
                        ax[_plot_idx].grid(True)

                        discriminator_score
                        fig.savefig("infoGAN_loss_plot_%d.png" % (epoch))
                        plt.close(fig)

                        fig, ax = plt.subplots(10, 10, figsize=(10, 10))
                        for c in range(0, 10):
                            for r in range(0, 10):
                                index = c * 10 + r
                                sub_plot = ax[c, r]
                                sub_plot.imshow(generated[index],
                                                cmap='gray', vmin=-1, vmax=1)
                                sub_plot.set_yticks([])
                                sub_plot.set_xticks([])
                        fileName = "infoGAN-%d.png" % (epoch)
                        fig.savefig(fileName, dpi=600)
                        return


                    if plot_process is not None:
                        plot_process.join()
                    if platform.system() == "Linux":
                        plot_process = multiprocessing.Process(target=_plot, args=(
                            generated_images, trainCnt, d_loss_log, g_loss_log,))
                        plot_process.start()
                    elif platform.system() == "Windows":
                        _plot(generated_images, trainCnt, d_loss_log, g_loss_log)
                    else:
                        print("!!! should not be here !!!")

    # After training, you can use the generator to generate new samples:
    generated_samples = generator.predict([np.random.normal(0, 1, (10, latent_dim)),
                                           np.random.uniform(-1, 1,
                                                             (10, num_continuous)),
                                           tf.keras.utils.to_categorical(np.arange(10), num_categories)])
