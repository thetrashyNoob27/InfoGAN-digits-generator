from tensorflow import keras
from tensorflow.keras import layers, Model


class Simple_Generator(Model):
    def __init__(self):
        super().__init__()
        self.main = keras.Sequential(
            [
                keras.Input(shape=(128,)),
                layers.Dense(7 * 7 * 128),
                layers.Reshape((7, 7, 128)),
                layers.Conv2DTranspose(128, kernel_size=4,
                                       strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, kernel_size=4,
                                       strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(1, (7, 7), padding="same", activation="tanh"),
            ],
            name="generator",
        )

    def call(self, x):
        return self.main(x)


class Simple_Discriminator(Model):
    def __init__(self):
        super().__init__()
        self.main = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.GlobalMaxPooling2D(),
                layers.Dense(1),
            ],
            name="discriminator",
        )

    def call(self, x):
        return self.main(x)


def my_Conditional_Generator(latent_dim, label_dim):
    import tensorflow as tf
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
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(1, (7, 7), padding="same", activation="tanh", name="generated")(x)

    model = tf.keras.Model(inputs=[latent_code_input, label_input], outputs=x)

    return model


class Conditional_Generator(Model):
    def __init__(self):
        super().__init__()

        self.main_linear1 = layers.Dense(7 * 7 * 128)
        self.main_reshape = layers.Reshape((7, 7, 128))
        self.main_conv2d_tr1 = layers.Conv2DTranspose(128, kernel_size=4,
                                                      strides=2,
                                                      padding="same")
        self.main_leaky1 = layers.LeakyReLU(alpha=0.2)
        self.main_conv2d_tr2 = layers.Conv2DTranspose(128, kernel_size=4,
                                                      strides=2,
                                                      padding="same")
        self.main_leaky2 = layers.LeakyReLU(alpha=0.2)
        self.main_conv2d_tr3 = layers.Conv2DTranspose(1, (7, 7),
                                                      padding="same",
                                                      activation="tanh")

        self.label_emb = layers.Embedding(input_dim=10, output_dim=30)
        self.label_linear1 = layers.Dense(units=7 * 7)
        self.label_reshape = layers.Reshape(target_shape=(7, 7, -1))

        self.concat_layer = layers.Concatenate(axis=-1)

    def call(self, latent_code, label):
        # turn label into activation map
        label_map = self.label_reshape(
            self.label_linear1(self.label_emb(label)))

        # turn latent code into activation map
        img_activation_maps = self.main_reshape(self.main_linear1(latent_code))

        # concatenate all maps
        concat_activation_maps = self.concat_layer(
            [img_activation_maps, label_map])

        # upsample to create image
        gen_img = self.main_conv2d_tr1(concat_activation_maps)
        gen_img = self.main_leaky1(gen_img)
        gen_img = self.main_conv2d_tr2(gen_img)
        gen_img = self.main_leaky2(gen_img)
        gen_img = self.main_conv2d_tr3(gen_img)

        return gen_img


class Conditional_Discriminator(Model):
    def __init__(self):
        super().__init__()

        self.main_conv2d_1 = layers.Conv2D(64, (4, 4),
                                           strides=(2, 2), padding="same")
        self.main_leaky_1 = layers.LeakyReLU(alpha=0.2)
        self.main_conv2d_2 = layers.Conv2D(128, (4, 4),
                                           strides=(2, 2), padding="same")
        self.main_leaky_2 = layers.LeakyReLU(alpha=0.2)
        self.main_conv2d_3 = layers.Conv2D(128, (4, 4),
                                           strides=(2, 2), padding="same")
        self.main_leaky_3 = layers.LeakyReLU(alpha=0.2)
        self.main_maxpool = layers.GlobalMaxPooling2D()
        self.main_linear = layers.Dense(1)

        self.label_emb = layers.Embedding(input_dim=10, output_dim=50)
        self.label_linear1 = layers.Dense(units=28 * 28)
        self.label_reshape = layers.Reshape(target_shape=(28, 28, -1))

        self.concat_layer = layers.Concatenate(axis=-1)

    def call(self, img, label):
        # turn label into activation map
        label_map = self.label_reshape(
            self.label_linear1(self.label_emb(label)))

        # concatenate all maps
        concat_img = self.concat_layer([img, label_map])

        # downsample to predict label
        x = self.main_conv2d_1(concat_img)
        x = self.main_leaky_1(x)
        x = self.main_conv2d_2(x)
        x = self.main_leaky_2(x)
        x = self.main_conv2d_3(x)
        x = self.main_leaky_3(x)
        x = self.main_maxpool(x)
        pred = self.main_linear(x)

        return pred


class ACGAN_Discriminator(Model):
    def __init__(self):
        super().__init__()

        self.main_conv2d_1 = layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same")
        self.main_leaky_1 = layers.LeakyReLU(alpha=0.2)
        self.main_conv2d_2 = layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same")
        self.main_leaky_2 = layers.LeakyReLU(alpha=0.2)
        self.main_conv2d_3 = layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same")
        self.main_leaky_3 = layers.LeakyReLU(alpha=0.2)
        self.main_maxpool = layers.GlobalMaxPooling2D()
        self.fake_head = layers.Dense(1)
        self.class_head = layers.Dense(10)

    def call(self, img):
        # downsample to predict label and class
        x = self.main_conv2d_1(img)
        x = self.main_leaky_1(x)
        x = self.main_conv2d_2(x)
        x = self.main_leaky_2(x)
        x = self.main_conv2d_3(x)
        x = self.main_leaky_3(x)
        x = self.main_maxpool(x)

        fake_pred = self.fake_head(x)
        class_pred = self.class_head(x)

        return fake_pred, class_pred
