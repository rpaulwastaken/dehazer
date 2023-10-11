import cv2
import numpy as np
import tensorflow as tf
from detect import hog_human_detection
from preprocess import preprocess


# Function to predict and visualize a single dehazed image
def predict_and_visualize(model, img):
    with tf.device('/GPU:0'):
        # Preprocess and resize the test image
        img = cv2.resize(img, (256, 256))
        cv2.imshow('Raw', img)
        img = preprocess(img)
        img = img.astype(np.float32) / 255.0

        # Resize the prediction
        prediction = model.predict(np.expand_dims(img, axis=0))[0]

        cv2.imshow('Prediction', prediction)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Function to read a video, apply de-hazing, and visualize human detection
def vid_read(model, directory):
    cap = cv2.VideoCapture(directory)

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        frame, rectangles = hog_human_detection(frame)
        frame = preprocess(frame)
        frame = cv2.resize(frame, (256, 256))
        frame = frame.astype(np.float32) / 255.0

        prediction = model.predict(np.expand_dims(frame, axis=0))[0]

        cv2.putText(prediction, f'Total Humans: {len(rectangles)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 2)

        cv2.imshow('frame', prediction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
