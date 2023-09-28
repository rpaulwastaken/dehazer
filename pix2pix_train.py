import tensorflow as tf
from pix2pix import generator_model, generator_loss, discriminator_loss, discriminator_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
import os

with tf.device('/CPU:0'):
    def train_pix2pix(train_dataset, val_dataset, output_dir, num_epochs=60, batch_size=3, model_name='model_name.h5', lambda_val=100):
        # Create the generator and discriminator U-Net
        generator = generator_model()
        discriminator = discriminator_model(input_shape=(256, 256, 3))

        # Define optimizers for generator and discriminator
        generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

        # Define loss metrics
        gen_loss_metric = Mean()
        disc_loss_metric = Mean()

        for epoch in range(num_epochs):
            gen_loss_metric.reset_states()
            disc_loss_metric.reset_states()

            for batch in train_dataset:
                input_image, target_image = batch

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    # Generate clear image using generator
                    gen_output = generator(input_image, training=True)

                    desired_input_shape = (256, 256)
                    resized_image = tf.image.resize(gen_output, desired_input_shape, method=tf.image.ResizeMethod.BICUBIC)

                    # Discriminator outputs for real and generated images
                    disc_real_output = discriminator([input_image, target_image], training=True)
                    disc_generated_output = discriminator([input_image, resized_image], training=True)

                    # Calculate losses
                    gen_loss, gan_loss, l1_loss = generator_loss(disc_generated_output, resized_image, target_image, lambda_val)
                    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

                # Calculate gradients and update weights
                gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
                disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

                # Update loss metrics
                gen_loss_metric(gen_loss)
                disc_loss_metric(disc_loss)

            # Evaluate the model on the validation dataset
            val_loss = evaluate(generator, val_dataset, discriminator, lambda_val)

            # Print and log the losses
            print(f"Epoch {epoch + 1}/{num_epochs}, Generator Loss: {gen_loss_metric.result()}, Discriminator Loss: {disc_loss_metric.result()}, Validation Loss: {val_loss}")

        # Save the final trained model
        generator.save(os.path.join(output_dir, model_name))


    def evaluate(generator, val_dataset, discriminator, lambda_val):
        # Define loss objects
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Loss metrics
        val_loss_metric = tf.keras.metrics.Mean()

        # Iterate through the validation dataset
        for input_image, target_image in val_dataset:
            # Generate an output image using the generator
            gen_output = generator(input_image, training=False)

            # Resize gen_output using bilinear interpolation
            gen_output_resized = tf.image.resize(gen_output, (256, 256), method=tf.image.ResizeMethod.BICUBIC)

            # Pass input_image and resized gen_output to the discriminator
            disc_generated_output = discriminator([input_image, gen_output_resized], training=False)

            # Resize target_image using bilinear interpolation
            target_image_resized = tf.image.resize(target_image, (256, 256), method=tf.image.ResizeMethod.BICUBIC)

            # Pass the target_image_resized to the discriminator to get disc_real_output
            disc_real_output = discriminator([input_image, target_image_resized], training=False)

            # Calculate generator loss
            gen_loss, _, _ = generator_loss(disc_generated_output, gen_output, target_image, lambda_val)

            # Calculate discriminator loss (make sure to get disc_real_output from somewhere)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            # Calculate total validation loss
            total_loss = gen_loss + disc_loss

            # Update loss metric
            val_loss_metric.update_state(total_loss)

        # Get the final validation loss
        final_val_loss = val_loss_metric.result()

        return final_val_loss
