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


def build_generator(latent_dim, num_categories):
    noise = tf.keras.layers.Input(shape=(latent_dim,))
    category_input = tf.keras.layers.Input(shape=(num_categories,))

    x = tf.keras.layers.Concatenate()(
        [noise, category_input])
    gen = x

    gen = tf.keras.layers.Dense(1024)(gen)
    gen = tf.keras.layers.BatchNormalization()(gen)
    gen = tf.keras.layers.LeakyReLU(alpha=0.1)(gen)

    n_nodes = 512 * 7 * 7
    gen = tf.keras.layers.Dense(n_nodes)(gen)
    gen = tf.keras.layers.BatchNormalization()(gen)
    gen = tf.keras.layers.LeakyReLU(alpha=0.1)(gen)
    gen = tf.keras.layers.Reshape((7, 7, 512))(gen)
    gen = tf.keras.layers.Dropout(0.05)(gen)
    # normal
    gen = tf.keras.layers.Conv2D(128, (4, 4), padding='same')(gen)
    gen = tf.keras.layers.BatchNormalization()(gen)
    gen = tf.keras.layers.LeakyReLU(alpha=0.1)(gen)
    gen = tf.keras.layers.Dropout(0.05)(gen)
    # upsample to 14x14
    gen = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = tf.keras.layers.BatchNormalization()(gen)
    gen = tf.keras.layers.LeakyReLU(alpha=0.1)(gen)
    gen = tf.keras.layers.Dropout(0.05)(gen)
    # upsample to 28x28
    gen = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same')(gen)
    # tanh output
    generated_image = tf.keras.layers.Activation('tanh', name="generator_output")(gen)

    return tf.keras.Model(inputs=[noise, category_input], outputs=generated_image)


def build_discriminator(input_shape):
    dinput = tf.keras.layers.Input(shape=(input_shape))
    x = dinput

    # downsample to 14x14
    d = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.1)(d)
    # downsample to 7x7
    d = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.1)(d)

    # flatten feature maps
    d = tf.keras.layers.Flatten()(d)
    d = tf.keras.layers.Dense(100)(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.1)(d)
    mid = d

    discrimate = tf.keras.layers.Dense(1, activation='sigmoid', name="D_discrimination")(mid)
    category_output = tf.keras.layers.Dense(10, activation='softmax', name="D_classify")(mid)

    d_model = tf.keras.Model(inputs=dinput, outputs=[discrimate, category_output])
    # compile model
    opt = tf.keras.optimizers.Adam()
    d_model.compile(loss={"D_discrimination": tf.keras.losses.BinaryCrossentropy(), "D_classify": tf.keras.losses.CategoricalCrossentropy()}, optimizer=opt)
    return d_model


def build_gan(g_model, d_model):
    for layer in d_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    gan_output = d_model(g_model.output)
    model = tf.keras.Model(inputs=g_model.input, outputs=gan_output)
    opt = tf.keras.optimizers.Adam()
    model.compile(loss=[tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.CategoricalCrossentropy()], optimizer=opt)
    return model


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
    latent_dim = 60
    num_categories = 10
    generator = build_generator(latent_dim, num_categories)
    discriminator = build_discriminator((28, 28, 1))
    gan = build_gan(generator, discriminator)
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
    next_report_time = time.monotonic() + REPORT_PERIOD_SEC
    d_loss_log = []
    g_loss_log = []
    d_score_log = []
    g_score_log = []
    trainCnt = 0
    discriminator_optmizer = tf.keras.optimizers.Adam()
    generator_optmizer = tf.keras.optimizers.SGD()
    for epoch in range(epochs):
        indices = np.arange(0, x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y = y[indices]
        x_batchs = np.array_split(x_train, np.ceil(x_train.shape[0] / batch_size))
        y_batchs = np.array_split(y, np.ceil(x_train.shape[0] / batch_size))
        batch_cnt = len(y_batchs)
        for idx in range(0, batch_cnt):
            trainCnt += 1

            real_images = x_batchs[idx]
            real_label = y_batchs[idx]
            real_label_oneHot = tf.keras.utils.to_categorical(real_label, num_categories)

            real_batch_size = real_images.shape[0]

            noise = np.random.normal(0, 1, (real_batch_size, latent_dim))
            sampled_categories = np.random.randint(0, num_categories, real_batch_size)
            sampled_categories_one_hot = tf.keras.utils.to_categorical(sampled_categories, num_categories)
            del sampled_categories
            valid = np.ones((real_batch_size, 1))
            fake = np.zeros((real_batch_size, 1))

            testoutput = discriminator(real_images, training=False)

            d_loss_real = discriminator.train_on_batch(real_images, [valid, real_label_oneHot], return_dict=True)['loss']
            fake_data = generator([noise, sampled_categories_one_hot], training=False)
            d_loss_fake = discriminator.train_on_batch(fake_data, [fake, sampled_categories_one_hot], return_dict=True)['loss']
            gan_loss = gan.train_on_batch([noise, sampled_categories_one_hot], [valid, sampled_categories_one_hot], return_dict=True)['loss']
            d_loss = np.mean([d_loss_real, d_loss_fake])

            d_loss_log.append(d_loss)
            g_loss_log.append(gan_loss)

            fake_discrime, _ = discriminator(fake_data, training=False)
            g_score = np.mean(fake_discrime)
            real_discrime, _ = discriminator(real_images, training=False)
            real_discrime = np.mean(real_discrime)
            d_score = np.mean([1 - g_score, real_discrime])
            del fake_discrime
            del _
            del real_discrime

            g_score_log.append(g_score)
            d_score_log.append(d_score)

            print("[%6d] d_loss:%10.4f(r%10.4f,f%10.4f) g_loss:%10.4f d_score:%10.4f g_score:%10.4f" % (trainCnt, d_loss, d_loss_real, d_loss_fake, gan_loss, d_score, g_score))

            # Print progress
            plot_process = None
            now_monotonic_time = time.monotonic()
            if now_monotonic_time > next_report_time:
                next_report_time += REPORT_PERIOD_SEC

                # plot image process
                noise = np.random.normal(0, 1, (100, latent_dim))
                cat_array = np.zeros((100, num_categories))
                for i in range(0, 100):
                    idx = i // num_categories
                cat_array[i, idx] = 1
                sampled_categories = cat_array
                del cat_array

                generated_images = generator.predict([noise, sampled_categories], verbose=0)


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
                    ax[_plot_idx].set_ylim(0, 1)
                    ax[_plot_idx].set_xlabel("epoch/tick")
                    ax[_plot_idx].set_ylabel('score')
                    ax[_plot_idx].set_title('D-G score')
                    ax[_plot_idx].legend()
                    ax[_plot_idx].grid(True)

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
