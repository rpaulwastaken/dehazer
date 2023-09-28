from data import load_and_split
from unet_train import train_unet
from pix2pix_train import train_pix2pix
from predict import predict_and_visualize, vid_read
import cv2
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    with tf.device('/CPU:0'): # Change this to '/GPU:0' if you have a GPU with enough memory
        # Specify the path to the hazy and ground truth images
        hazy_dir = 'dataset/Hazy_Indoor'
        gt_dir = 'dataset/GT_Indoor'

        # Load, split, and preprocess the dataset for training
        X_train, X_val, y_train, y_val, input_shape = load_and_split(hazy_dir, gt_dir)

        # reduce this if you run out of memory
        batch_size = 8

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        # Train the pix2pix model
        output_dir = 'models/pix2pix'
        train_pix2pix(train_dataset, val_dataset, output_dir, num_epochs=15, batch_size=batch_size, model_name='p2p.h5')

        # Uncomment the following to train the U-Net model
        # output_dir = 'models/U-Net'
        # train_unet(X_train, X_val, y_train, y_val, input_shape, output_dir, num_epochs=15, batch_size=batch_size, model_name='unet.h5')

        # Specify the path to the trained dehazing model
        dehaze_model = 'models/pix2pix/p2p.h5'

        # Load the trained dehazing model
        model = tf.keras.models.load_model(dehaze_model)

        # Specify the path to the input video file for human detection and dehazing
        directory = 'test_files/test_vid.mp4'

        # Call the function to read the video, perform human detection, and dehazing
        with tf.device('/GPU:0'):
            vid_read(model, directory)

        # Specify the path to a test hazy image for dehazing
        test_img_dir = 'test_files/hazedimg.jpg'
        img = cv2.imread(test_img_dir)

        # Predict and visualize the dehazed image
        predict_and_visualize(model, img)
