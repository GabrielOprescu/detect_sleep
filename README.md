# detect_sleep

Train a model to detect if the eyes are closed or opened.

Usefull to tell if a person sleeps.

### 1
First I tried to develop a model that draws boxes around the eyes. Because an object detection model is resource and computational expensive, I used a feature extractor to train a model that outputs 4 values, one for each corner of a rectangle draw around the eyes. The results were not grate.
https://github.com/GabrielOprescu/detect_sleep/blob/master/eye_box_model.py


### 2
Having a dataset with eyes in low resolution and the label opened or closed, I used a pretrained model with the last layer modified to predict it an eye is opened or closed. 
https://github.com/GabrielOprescu/detect_sleep/blob/master/open_closed_eye_model.py


### 3
To extract the eyes from the video, I used haar cascade procedure available in open cv. With the model trained earlier, per each eye a prediction of open or closed was made. If both eyes are close for a certain period of time, on the video a woarnnign message appears.
https://github.com/GabrielOprescu/detect_sleep/blob/master/find_eyes.py

For the bet performance be sore to have good lighting on the face when video starts.
