import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Concatenate, UpSampling2D, Flatten, Dense, MaxPooling2D, concatenate
from tensorflow.keras.models import Model

with tf.device('/CPU:0'):
    def generator_model():
        input_shape = (256, 256, 3)
        inputs = Input(shape=input_shape)

        # encoder
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

        # Decoder
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)

        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)

        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)

        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)

        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[outputs])

        return model


    def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # GAN loss
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        gen_output_resized = tf.image.resize(gen_output, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Mean absolute error (L1 loss)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output_resized))

        # Total generator loss
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss


    def discriminator_model(input_shape):
        input_image = Input(shape=input_shape)
        target_image = Input(shape=input_shape)

        # Concatenate input_image and target_image along the channel dimension
        concatenated_image = Concatenate()([input_image, target_image])

        x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(concatenated_image)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(1, (4, 4), strides=(1, 1), padding='same')(x)

        x = Flatten()(x)
        x = Dense(1)(x)

        model = Model(inputs=[input_image, target_image], outputs=x)

        return model


    def discriminator_loss(disc_real_output, disc_generated_output):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # Real loss
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

        # Generated loss
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        # Total discriminator loss
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
