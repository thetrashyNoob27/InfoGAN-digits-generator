import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# generator model is the same with Cond-GAN
from model import Conditional_Generator, ACGAN_Discriminator, my_Conditional_Generator


def train(cfg, train_ds):
    # models
    net_g, net_d = my_Conditional_Generator(128, 10), ACGAN_Discriminator()

    # loss functions
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, label_smoothing=cfg['label_smoothing'])
    aux_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    # optimizers
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['g_lr'])
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['d_lr'])

    # metrics
    disc_loss_tracker = tf.keras.metrics.Mean(name='disc_loss')
    gen_loss_tracker = tf.keras.metrics.Mean(name='gen_loss')

    # tensorboard
    summary_writer = tf.summary.create_file_writer(
        os.path.join(
            cfg['log_dir'], cfg['model'], cfg['experiment_name']))

    latent_code_size = cfg['latent_code_size']

    # fix latent code to track improvement
    latent_code4visualization = tf.random.normal(shape=(25, latent_code_size))
    labels4visualization = np.concatenate(
        (np.repeat(np.arange(start=0, stop=10), 2), np.array([2, 3, 4, 5, 6])))

    for epoch in range(cfg['epochs']):

        for _, (real_imgs, real_lbls) in train_ds.enumerate():
            # PART 1: DISC TRAINING, fixed generator
            latent_code = tf.random.normal(shape=(real_imgs.shape[0],
                                                  latent_code_size))

            with tf.GradientTape() as disc_tape:
                # generate fake images
                generated_imgs = net_g([latent_code, real_lbls])

                # forward pass real and fake images
                real_preds, real_class_preds = net_d(real_imgs)
                fake_preds, fake_class_preds = net_d(generated_imgs)

                y_pred = tf.concat([real_preds, fake_preds], axis=0)
                y_true = tf.concat([tf.ones_like(real_preds),
                                    tf.zeros_like(fake_preds)], axis=0)

                y_pred_class = tf.concat(
                    [real_class_preds, fake_class_preds], axis=0)
                y_true_class = tf.concat([real_lbls, real_lbls], axis=0)

                # compute loss
                disc_fake_loss = loss_fn(y_true=y_true, y_pred=y_pred)
                disc_aux_loss = aux_loss_fn(
                    y_true=y_true_class, y_pred=y_pred_class)
                disc_loss = disc_fake_loss + disc_aux_loss

            # compute disc gradients
            disc_gradients = disc_tape.gradient(disc_loss,
                                                net_d.trainable_variables)

            # update disc weights
            disc_optimizer.apply_gradients(zip(disc_gradients,
                                               net_d.trainable_variables))

            # update disc metrics
            disc_loss_tracker.update_state(disc_loss)

            # PART 2: GEN TRAINING, fixed discriminator
            latent_code = tf.random.normal(shape=(real_imgs.shape[0],
                                                  latent_code_size))

            with tf.GradientTape() as gen_tape:
                # generate fake images
                generated_imgs = net_g([latent_code, real_lbls])

                # forward pass only images
                fake_preds, fake_class_preds = net_d(generated_imgs)

                # compute loss
                gen_fake_loss = loss_fn(y_true=tf.ones_like(fake_preds),
                                        y_pred=fake_preds)
                gen_aux_loss = aux_loss_fn(y_true=real_lbls,
                                           y_pred=fake_class_preds)
                gen_loss = gen_fake_loss + gen_aux_loss

            # compute gen gradients
            gen_gradients = gen_tape.gradient(gen_loss,
                                              net_g.trainable_variables)

            # update gen weights
            gen_optimizer.apply_gradients(zip(gen_gradients,
                                              net_g.trainable_variables))

            # update gen metrics
            gen_loss_tracker.update_state(gen_loss)

            #net_g.save("ACGAN-model-G.h5", save_format="h5")
            #net_d.save("ACGAN-model-D.tf", save_format="tf")

        # generate and save sample images per epoch
        test_generated_imgs = net_g([latent_code4visualization, labels4visualization])
        test_generated_imgs = (((test_generated_imgs + 1.) / 2.) * 255.).numpy()
        plt.figure(figsize=(5, 5))
        for i in range(test_generated_imgs.shape[0]):
            plt.subplot(5, 5, i + 1)
            plt.imshow(test_generated_imgs[i, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.savefig(os.path.join(
            cfg['img_save_dir'], cfg['model'], cfg['experiment_name']))
        plt.close()

        # display and record metrics at the end of each epoch.
        with summary_writer.as_default():
            tf.summary.scalar('disc_loss', disc_loss_tracker.result(),
                              step=epoch)
            tf.summary.scalar('gen_loss', gen_loss_tracker.result(),
                              step=epoch)
            tf.summary.image(name='test_samples', data=test_generated_imgs,
                             max_outputs=test_generated_imgs.shape[0],
                             step=epoch)

        disc_loss = disc_loss_tracker.result()
        gen_loss = gen_loss_tracker.result()
        print(f'epoch: {epoch}, disc_loss: '
              f'{disc_loss:.4f}, gen_loss: {gen_loss:.4f}')

        # reset metric states
        disc_loss_tracker.reset_state()
        gen_loss_tracker.reset_state()

    return
