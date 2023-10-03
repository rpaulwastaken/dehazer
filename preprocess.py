import cv2
import tensorflow as tf


# Function to preprocess an image for de-hazing
def preprocess(frame):
    with tf.device('/CPU:0'):
        # Increase contrast on the image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        dehazed_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Further adjustments in the HLS color space
        hls = cv2.cvtColor(dehazed_frame, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        l = cv2.subtract(l, 40)
        s = cv2.add(s, 30)
        hls = cv2.merge((h, l, s))
        final_frame = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

        return final_frame
