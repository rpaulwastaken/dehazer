import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess import preprocess


# Function to load and preprocess a dataset of hazy and ground truth images
def load_and_preprocess_images(hazy_dir, gt_dir, target_size=(256, 256)):
    with tf.device('/CPU:0'):
        hazy_files = os.listdir(hazy_dir)
        hazy_images = []  # List to store preprocessed hazy images
        truth_images = []  # List to store preprocessed ground truth images

        for hazy_file in hazy_files:
            if hazy_file.endswith('.jpg') or hazy_file.endswith('.png'):
                # Load the hazy image
                hazy_path = os.path.join(hazy_dir, hazy_file)
                hazy_image = cv2.imread(hazy_path)

                # Preprocess the hazy image
                hazy_image = preprocess(hazy_image)

                # Resize the hazy image to the target size (e.g., 256x256)
                hazy_image = cv2.resize(hazy_image, target_size)

                # Normalize the pixel values to the range [0, 1]
                hazy_image = hazy_image.astype(np.float32) / 255.0

                # Find the corresponding ground truth image based on naming convention
                gt_file = hazy_file.replace('_hazy.png', '_GT.png')  # Assuming ground truths have a naming convention
                gt_path = os.path.join(gt_dir, gt_file)

                if os.path.exists(gt_path):
                    # Load the corresponding ground truth image
                    ground_truth = cv2.imread(gt_path)

                    # Resize the ground truth image to the target size (e.g., 256x256)
                    ground_truth = cv2.resize(ground_truth, target_size)

                    # Normalize the pixel values to the range [0, 1]
                    ground_truth = ground_truth.astype(np.float32) / 255.0

                    # Append the preprocessed images to their respective lists
                    hazy_images.append(hazy_image)
                    truth_images.append(ground_truth)

        return hazy_images, truth_images


# Function to load and split a dataset into training and validation sets
def load_and_split(hazy_dir, gt_dir):
    # Get the raw sizes of the hazy images
    raw_size = []
    for hazy_file in os.listdir(hazy_dir):
        if hazy_file.endswith('.jpg') or hazy_file.endswith('.png'):
            hazy_path = os.path.join(hazy_dir, hazy_file)
            hazy_image = cv2.imread(hazy_path)
            raw_size.append(hazy_image.shape)

    # Load and preprocess hazy images and their corresponding ground truths
    hazy_images, raw_images = load_and_preprocess_images(hazy_dir, gt_dir)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(hazy_images, raw_images, test_size=0.2, random_state=42)

    # Convert the lists to numpy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Get the input shape
    input_shape = (256, 256, 3)

    return X_train, X_val, y_train, y_val, input_shape
