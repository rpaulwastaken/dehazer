import tensorflow as tf
from model import sarra


# Function to train a dehazing model
def train_unet(X_train, X_val, y_train, y_val, input_shape, output_dir, num_epochs=60, batch_size=3, model_name='my_Model.h5'):
    with tf.device('/GPU:0'):
        # Define the dehazing model
        model = sarra(input_shape)

        # Compile the model with an appropriate loss function
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, batch_size=batch_size)

        # Save the trained dehazing model
        model.save(output_dir + model_name)
