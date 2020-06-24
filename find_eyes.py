import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime

# not use now
class SecondCounter:

    def __init__(self):
        self.start_now = None

    def start(self):
        self.start_now = datetime.now()

    @property
    def value(self):
        return (datetime.now() - self.start_now).seconds


# Initiate paths

blink_model_path = './open_closed_eye_model'
class_names = ['close', 'open']
eye_xml_path = './haarcascade_eye.xml'

# Load models

eye_detector = cv2.CascadeClassifier(eye_xml_path)
blink_model = tf.keras.models.load_model(blink_model_path)


# blink_model.summary()


# Define main function

def main():
    capture = cv2.VideoCapture(0)
    minNeighbors = 8  # is the best one. can be adjusted
    scaleFactor = 1.2
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(f'The video is runing at {fps} frames per second')

    eye_status = 'close'
    new_eye_status = 'close'

    frame_counter = 0
    new_frame_counter = 0
    constant_counter = 0

    warn_msg = 'All good'
    warn_find = 'Eyes were find'

    # start teh counter
    count = SecondCounter()
    count.start()

    while True:

        # get the seconds passed

        # capture the frame
        ret, frame = capture.read()

        # transform image to gray scale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # print('MinN = ', minNeighbors, 'ScaF = ', scaleFactor)

        # use the harr cascade to detect eye location
        eyes = eye_detector.detectMultiScale(gray_frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                             minSize=(5, 5), maxSize=(200, 200))

        # for each detected eye apply the open closes model
        for i, (x, y, w, h) in enumerate(eyes):

            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(100, 100, 100), thickness=2, lineType=cv2.LINE_4)

            eye_img = frame[y:(y + h), x:(x + w)]
            eye_img = cv2.resize(eye_img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
            eye_img = tf.image.convert_image_dtype(eye_img, tf.float32)

            pred = blink_model.predict(eye_img[np.newaxis, ...])
            new_eye_status = class_names[np.argmax(pred)]


            if eye_status == 'close' and new_eye_status == 'close':
                frame_counter += 1
                if frame_counter > 15:
                    warn_msg = 'Either we cannot find your eyes or you are sleeping'
            elif eye_status == 'open' and new_eye_status == 'close':
                frame_counter = 0
            else:
                warn_msg = 'All good'

            print('Frame_counter: ', frame_counter)

            cv2.putText(frame, eye_status, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(255, 255, 255))

            # cv2.imshow('eye'+str(i), eye_img)

        if new_frame_counter == frame_counter:
            constant_counter += 1
            if constant_counter > 15:
                warn_find = 'No eyes found'
        else:
            constant_counter = 0
            warn_find = 'Eyes were find'

        new_frame_counter = frame_counter

        print('Constant counter: ', constant_counter)

        # insert the counter
        cv2.putText(frame, warn_msg, org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255))
        cv2.putText(frame, warn_find, org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255))

        cv2.imshow('Frame', frame)

        eye_status = new_eye_status

        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
