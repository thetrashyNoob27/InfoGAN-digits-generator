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


def build_generator(latent_dim, label_dim):
    # input shape define
    latent_code_input = tf.keras.layers.Input(shape=(latent_dim,), name="latent_code")
    label_input = tf.keras.layers.Input(shape=(1,), name="label")

    # label encode
    label_encode = tf.keras.layers.Embedding(input_dim=label_dim, output_dim=30)(label_input)
    label_encode = tf.keras.layers.Dense(units=7 * 7)(label_encode)
    label_encode = tf.keras.layers.Reshape(target_shape=(7, 7, -1))(label_encode)

    # latent code encode
    latent_encode = tf.keras.layers.Dense(7 * 7 * 128)(latent_code_input)
    latent_encode = tf.keras.layers.Reshape((7, 7, 128))(latent_encode)

    # combine & gen image/data
    x = tf.keras.layers.Concatenate(axis=-1)([latent_encode, label_encode])
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(1, (7, 7), padding="same", activation="tanh", name="generated")(x)

    model = tf.keras.Model(inputs=[latent_code_input, label_input], outputs=x)

    return model


def build_discriminator():
    # input shape define
    d_input = tf.keras.layers.Input(shape=(28, 28, 1), name="discriminator_input")

    x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same")(d_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)

    mid = x
    discriminate = tf.keras.layers.Dense(1, activation="sigmoid")(mid)
    classify = tf.keras.layers.Dense(10, activation="softmax")(mid)

    model = tf.keras.Model(inputs=d_input, outputs=[discriminate, classify])

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


def gan_score(fake_discrime, real_discrime):
    g_score = np.mean(fake_discrime)
    real_discrime = np.mean(real_discrime)
    d_score = np.mean([1 - g_score, real_discrime])
    return g_score, d_score


if __name__ == "__main__":
    remove_old_image()
    if platform.system() == "Linux":
        os.nice(19)
    # Dimensions
    latent_dim = 60
    num_categories = 10
    generator = build_generator(latent_dim, num_categories)
    discriminator = build_discriminator()
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
    epochs = 30
    batch_size = 128

    # Training loop
    REPORT_PERIOD_SEC = 60
    next_report_time = time.monotonic() + REPORT_PERIOD_SEC
    d_loss_log = []
    g_loss_log = []
    d_score_log = []
    g_score_log = []
    trainCnt = 0
    discriminator_optmizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    generator_optmizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
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
            real_batch_size = real_images.shape[0]

            noise = np.random.normal(0, 1, (real_batch_size, latent_dim))
            sampled_categories = np.random.randint(0, num_categories, real_batch_size)

            valid = np.ones((real_batch_size, 1))
            fake = np.zeros((real_batch_size, 1))

            # train discriminator
            with tf.GradientTape() as d_tape:
                generated_data = generator([noise, sampled_categories])

                # forward pass real and fake images
                real_data_prediction, real_class_prediction = discriminator(real_images)
                fake_data_prediction, fake_class_prediction = discriminator(generated_data)

                y_pred = tf.concat([real_data_prediction, fake_data_prediction], axis=0)
                y_true = tf.concat([valid, fake], axis=0)

                y_pred_class = tf.concat([real_class_prediction, fake_class_prediction], axis=0)
                y_true_class = tf.concat([real_label, sampled_categories], axis=0)

                # compute loss
                disc_fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)(y_true=y_true, y_pred=y_pred)
                disc_aux_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true=y_true_class, y_pred=y_pred_class)
                disc_loss = disc_fake_loss + disc_aux_loss

                # score log
                g_score, d_score = gan_score(fake_data_prediction, real_data_prediction)
                d_score_log.append(d_score)
                g_score_log.append(g_score)
                d_loss_log.append(disc_loss)

                # compute disc gradients
            disc_gradients = d_tape.gradient(disc_loss, discriminator.trainable_variables)

            # update disc weights
            discriminator_optmizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # PART 2: GEN TRAINING, fixed discriminator
            noise = np.random.normal(0, 1, (real_batch_size, latent_dim))
            sampled_categories = np.random.randint(0, num_categories, real_batch_size)
            with tf.GradientTape() as g_tape:
                generated_data = generator([noise, sampled_categories])

                # forward pass only images
                fake_preds, fake_class_preds = discriminator(generated_data)

                # compute loss
                gen_fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)(y_true=tf.ones_like(fake_preds), y_pred=fake_preds)
                gen_aux_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true=sampled_categories, y_pred=fake_class_preds)
                gen_loss = gen_fake_loss + gen_aux_loss

                # loss logger
                g_loss_log.append(gen_loss)

            # compute gen gradients
            gen_gradients = g_tape.gradient(gen_loss, generator.trainable_variables)

            # update gen weights
            generator_optmizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

            # report functions
            print("[%6d] d_loss:%10.4f g_loss:%10.4f d_score:%10.4f g_score:%10.4f" % (trainCnt, d_loss_log[-1], g_loss_log[-1], d_score_log[-1], g_score_log[-1]))

            # Print progress
            plot_process = None
            now_monotonic_time = time.monotonic()
            if now_monotonic_time > next_report_time:
                next_report_time += REPORT_PERIOD_SEC

                # plot image process
                noise = np.random.normal(0, 1, (100, latent_dim))
                cat_array = np.zeros((100, 1))
                for i in range(0, 100):
                    idx = i // num_categories
                    cat_array[i, 0] = idx
                sampled_categories = cat_array
                del cat_array

                generated_images = generator.predict([noise, sampled_categories], verbose=0)


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
