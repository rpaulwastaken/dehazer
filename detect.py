import cv2


# Function to detect humans using HOG (Histogram of Oriented Gradients)
def hog_human_detection(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detect humans using HOG
    rectangles, weights = hog.detectMultiScale(frame)

    # Draw rectangles around detected humans
    for (x, y, w, h) in rectangles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, rectangles
