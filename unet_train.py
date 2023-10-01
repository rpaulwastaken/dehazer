import os.path
import tensorflow as tf
from unet_model import unet


# Function to train a de-hazing model
def train_unet(X_train, X_val, y_train, y_val, input_shape, output_dir, num_epochs=60, batch_size=3, model_name='my_Model.h5'):
    with tf.device('/GPU:0'):
        # Define the de-hazing model
        model = unet(input_shape)

        # Compile the model with an appropriate loss function
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, batch_size=batch_size)

        # Save the trained de-hazing model
        model.save(os.path.join(output_dir, model_name))
