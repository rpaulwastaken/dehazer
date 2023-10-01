from data import load_and_split
from unet_train import train_unet
from pix2pix_train import train_pix2pix
from predict import predict_and_visualize, vid_read
import cv2
import tensorflow as tf

if __name__ == '__main__':
    with tf.device('/CPU:0'):  # Change this to '/GPU:0' if you have a GPU with enough memory
        # Specify the path to the hazy and ground truth images
        hazy_dir = 'dataset/Hazy_Indoor'
        gt_dir = 'dataset/GT_Indoor'

        # Load, split, and preprocess the dataset for training
        X_train, X_val, y_train, y_val, input_shape = load_and_split(hazy_dir, gt_dir)

        # reduce this if you run out of memory
        batch_size = 8

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        # Train the Pix2Pix model
        output_dir = 'models/pix2pix'
        train_pix2pix(train_dataset,
                      val_dataset,
                      output_dir,
                      num_epochs=15,
                      batch_size=batch_size,
                      model_name='p2p.h5')

        # Uncomment the following to train the U-Net model
        # output_dir = 'models/U-Net'
        # train_unet(X_train, X_val,
        #           y_train, y_val,
        #           input_shape,
        #           output_dir,
        #           num_epochs=15,
        #           batch_size=batch_size,
        #           model_name='unet.h5')

        # Specify the path to the trained de-hazing model
        dehaze_model = 'models/pix2pix/p2p.h5'

        # uncomment the following to load the U-Net model instead
        # dehaze_model = 'models/U-Net/unet.h5'

        # Load the trained de-hazing model
        model = tf.keras.models.load_model(dehaze_model)

        # Specify the path to the input video file for human detection and de-hazing
        directory = 'sample_test_files/test_vid.mp4'

        # Call the function to read the video, perform human detection, and de-hazing
        with tf.device('/GPU:0'):
            vid_read(model, directory)

        # Specify the path to a test hazy image for de-hazing
        test_img_dir = 'sample_test_files/hazed_img.jpg'
        img = cv2.imread(test_img_dir)

        # Predict and visualize the de-hazed image
        predict_and_visualize(model, img)
