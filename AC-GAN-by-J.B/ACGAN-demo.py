#!/usr/bin/env python3
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot
import time
import numpy as np
import platform
import multiprocessing

import matplotlib.pyplot as plt


# define the standalone discriminator model
def define_discriminator(in_shape=(28, 28, 1), n_classes=10):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=in_shape)
    # downsample to 14x14
    fe = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # normal
    fe = Conv2D(64, (3, 3), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # downsample to 7x7
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # normal
    fe = Conv2D(256, (3, 3), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    # define model
    model = Model(in_image, [out1, out2])
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model


# define the standalone generator model
def define_generator(latent_dim, n_classes=10):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 7 * 7
    li = Dense(n_nodes, kernel_initializer=init)(li)
    # reshape to additional channel
    li = Reshape((7, 7, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 384 * 7 * 7
    gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
    gen = Activation('relu')(gen)
    gen = Reshape((7, 7, 384))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    # upsample to 14x14
    gen = Conv2DTranspose(192, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(merge)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
    out_layer = Activation('tanh')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if isinstance(layer, BatchNormalization):
            continue
        layer.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model


# load images
def load_real_samples():
    # load dataset
    (trainX, trainy), (_, _) = load_data()
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    print(X.shape, trainy.shape)
    return [X, trainy]


# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input], verbose=0)
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
    # prepare fake examples
    [X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(100):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename1 = 'generated_plot_%04d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%04d.h5' % (step + 1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs

    REPORT_PERIOD_SEC = 60
    next_report_time = time.monotonic() + REPORT_PERIOD_SEC
    d_loss_log = []
    g_loss_log = []
    d_score_log = []
    g_score_log = []
    trainCnt = 0
    plot_process = None
    for i in range(n_steps):
        trainCnt += 1
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        d_loss_real = d_model.train_on_batch(X_real, [y_real, labels_real], return_dict=True)['loss']
        # generate 'fake' examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        d_loss_fake = d_model.train_on_batch(X_fake, [y_fake, labels_fake], return_dict=True)['loss']
        # prepare points in latent space as input for the generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        gan_loss = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels], return_dict=True)['loss']
        # summarize loss on this batch
        d_loss = np.mean([d_loss_real, d_loss_fake])

        d_loss_log.append(d_loss)
        g_loss_log.append(gan_loss)

        fake_discrime, _ = discriminator(X_fake, training=False)
        g_score = np.mean(fake_discrime)
        real_discrime, _ = discriminator(X_real, training=False)
        real_discrime = np.mean(real_discrime)
        d_score = np.mean([1 - g_score, real_discrime])
        del fake_discrime
        del _
        del real_discrime

        g_score_log.append(g_score)
        d_score_log.append(d_score)

        print("[%6d] d_loss:%10.4f g_loss:%10.4f d_score:%10.4f g_score:%10.4f" % (trainCnt, d_loss, gan_loss, d_score, g_score))
        # evaluate the model performance every 'epoch'
        now_monotonic_time = time.monotonic()
        if now_monotonic_time > next_report_time:
            next_report_time += REPORT_PERIOD_SEC
            summarize_performance(i, g_model, latent_dim)

            def _plot(epoch, d_loss_log, g_loss_log):
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
                return

            if plot_process is not None:
                plot_process.join()
            if platform.system() == "Linux":
                plot_process = multiprocessing.Process(target=_plot, args=(
                    trainCnt, d_loss_log, g_loss_log,))
                plot_process.start()
            elif platform.system() == "Windows":
                _plot(trainCnt, d_loss_log, g_loss_log)
            else:
                print("!!! should not be here !!!")


if __name__ == "__main__":
    # size of the latent space
    latent_dim = 100
    # create the discriminator
    discriminator = define_discriminator()
    # create the generator
    generator = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(generator, discriminator)
    # load image data
    dataset = load_real_samples()
    # train model
    train(generator, discriminator, gan_model, dataset, latent_dim)
