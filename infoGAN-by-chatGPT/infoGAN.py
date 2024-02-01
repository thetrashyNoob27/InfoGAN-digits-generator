#!/usr/bin/env python3
# example of training an infogan on mnist
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import tensorflow as tf
np.random.seed(16)
tf.random.set_seed(16)


def build_generator(latent_dim, num_continuous, num_categories):
    noise = tf.keras.layers.Input(shape=(latent_dim,))
    continuous_input = tf.keras.layers.Input(shape=(num_continuous,))
    category_input = tf.keras.layers.Input(shape=(num_categories,))

    x = tf.keras.layers.Concatenate()(
        [noise, continuous_input, category_input])
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(784, activation='sigmoid')(x)
    generated_image = tf.keras.layers.Reshape((28, 28, 1))(x)

    return tf.keras.models.Model([noise, continuous_input, category_input], generated_image)


def build_discriminator(num_continuous, num_categories):
    image_input = tf.keras.layers.Input(shape=(28, 28, 1))
    flat_image = tf.keras.layers.Flatten()(image_input)

    x = tf.keras.layers.Dense(256, activation='relu')(flat_image)

    validity = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Auxiliary outputs
    continuous_output = tf.keras.layers.Dense(
        num_continuous, activation='linear')(x)
    category_output = tf.keras.layers.Dense(
        num_categories, activation='softmax')(x)

    return tf.keras.models.Model(image_input, [validity, continuous_output, category_output])


def mutual_information_loss(c, c_given_x):
    """Mutual information loss."""
    eps = 1e-8
    conditional_entropy = - \
        tf.reduce_mean(tf.reduce_sum(
            c_given_x * tf.math.log(c_given_x + eps), axis=1))
    entropy = -tf.reduce_mean(tf.reduce_sum(c * tf.math.log(c + eps), axis=1))
    return conditional_entropy + entropy


if __name__ == "__main__":
    # Dimensions
    latent_dim = 62
    num_continuous = 2
    num_categories = 10

    # Build and compile the generator, discriminator, and InfoGAN models
    generator = build_generator(latent_dim, num_continuous, num_categories)
    discriminator = build_discriminator(num_continuous, num_categories)

    discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss=[
                          'binary_crossentropy', 'mse', 'categorical_crossentropy'])

    # InfoGAN model
    try:
        info_gan_model = tf.keras.models.load_model("infoGAN-mode.tf")
        print("load model success")
    except OSError as e:
        noise = tf.keras.layers.Input(shape=(latent_dim,))
        continuous_input = tf.keras.layers.Input(shape=(num_continuous,))
        category_input = tf.keras.layers.Input(shape=(num_categories,))
        generated_image = generator([noise, continuous_input, category_input])
        discriminator.trainable = False
        validity, continuous_output, category_output = discriminator(
            generated_image)
        info_gan_model = tf.keras.models.Model([noise, continuous_input, category_input], [
            validity, continuous_output, category_output])

        info_gan_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                               loss=['binary_crossentropy', 'mse',
                                     'categorical_crossentropy'],
                               loss_weights=[1, 0.5, 1])
    # prepare dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    del x_train, y_train, x_test, y_test
    x_train = x / 255.0
    x_train = x_train.reshape(x_train.shape + (1,)).astype("float32")

    # Build and compile the generator, discriminator, and InfoGAN models
    # (As per the previous example code)

    # Training parameters
    epochs = 10000
    batch_size = 64
    half_batch = batch_size // 2

    # Training loop
    plot_process = None
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        real_images = x_train[idx]

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        sampled_categories = np.random.randint(0, num_categories, half_batch)
        sampled_categories_one_hot = tf.keras.utils.to_categorical(
            sampled_categories, num_categories)
        sampled_continuous = np.random.uniform(-1,
                                               1, (half_batch, num_continuous))

        generated_images = generator.predict(
            [noise, sampled_continuous, sampled_categories_one_hot], verbose=0)

        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch(
            real_images, [valid, sampled_continuous, sampled_categories_one_hot])
        d_loss_fake = discriminator.train_on_batch(
            generated_images, [fake, sampled_continuous, sampled_categories_one_hot])
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_categories = np.random.randint(0, num_categories, batch_size)
        sampled_categories_one_hot = tf.keras.utils.to_categorical(
            sampled_categories, num_categories)
        sampled_continuous = np.random.uniform(-1,
                                               1, (batch_size, num_continuous))

        valid = np.ones((batch_size, 1))

        g_loss = info_gan_model.train_on_batch([noise, sampled_continuous, sampled_categories_one_hot],
                                               [valid, sampled_continuous, sampled_categories_one_hot])

        # Print progress
        if epoch % 100 == 0 and epoch != 0:
            print(
                f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss[0]}]")

            # save model
            info_gan_model.save("infoGAN-mode.tf", save_format="tf")

            # plot image process
            noise = np.random.normal(0, 1, (100, latent_dim))
            cat_array = np.zeros((100, num_categories))
            for i in range(0, 100):
                idx = i//num_categories
                cat_array[i, idx] = 1
            sampled_categories = cat_array
            del cat_array

            sampled_continuous = np.concatenate(
                (np.linspace(-1, 1, 100).reshape((-1, 1)), np.linspace(-1, 1, 100).reshape((-1, 1))), axis=1)

            generated_images = generator.predict(
                [noise, sampled_continuous, sampled_categories], verbose=0)
            generated_images = generated_images*255
            generated_images = generated_images.astype(int)
            generated_images = 255-generated_images

            # print(generated_images.shape)#(100, 28, 28, 1)

            def _plot(generated, epoch):
                fig, ax = plt.subplots(10, 10, figsize=(10, 10))
                for c in range(0, 10):
                    for r in range(0, 10):
                        index = c * 10 + r
                        sub_plot = ax[c, r]
                        sub_plot.imshow(generated[index],
                                        cmap='gray', vmin=0, vmax=255)
                        sub_plot.set_yticks([])
                        sub_plot.set_xticks([])
                fileName = "infoGAN-%d.png" % (epoch)
                fig.savefig(fileName, dpi=600)
                return
            if plot_process is not None:
                plot_process.join()
                del plot_process
            plot_process = multiprocessing.Process(target=_plot, args=(
                generated_images, epoch,))
            plot_process.start()

    # After training, you can use the generator to generate new samples:
    generated_samples = generator.predict([np.random.normal(0, 1, (10, latent_dim)),
                                           np.random.uniform(-1, 1,
                                                             (10, num_continuous)),
                                           tf.keras.utils.to_categorical(np.arange(10), num_categories)])
