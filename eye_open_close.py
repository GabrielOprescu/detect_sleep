"""
Classifier to detect if an eye is close or opened.
It does not care if it is left or right eye.
"""

import numpy as np
import cv2
import tensorflow as tf
import pathlib
import os
import zipfile
import matplotlib.pyplot as plt
import random
import tensorflow_hub as hub


# read the archive with the eye images
with zipfile.ZipFile('dataset_B_Eye_Images.zip', mode='r') as zp:
    zp.extractall()


# set the directory where the images will be found
data_dir = pathlib.Path('dataset_B_Eye_Images')


# remove not jpg files
other_files = list(data_dir.glob('*/*.db'))

print(other_files)

for x in other_files:
    os.remove(str(x))


# take the folder names
fold_names = list(data_dir.glob('*/'))
print(fold_names)


# separate the folders by open or close. It is specified in the name of the folder

open = [fold for fold in fold_names if str.find(str(fold), 'open') != -1]
close = [fold for fold in fold_names if str.find(str(fold), 'closed') != -1]

print('Open:', open)
print('Closed: ', close)


# take a random choice of files that will go in the valid folder
rand_chose = {}

for fold in fold_names:
    pict = list(fold.glob('*.jpg'))
    rand_chose[fold] = random.choices(pict, k=10)


# create train and valid folder and inside them all the 2 folder above (open close) to have the same structure as parent
train_dir = pathlib.Path(data_dir, 'train')
valid_dir = pathlib.Path(data_dir, 'valid')

os.mkdir(str(pathlib.Path(data_dir, 'train')))
os.mkdir(str(pathlib.Path(data_dir, 'valid')))

for f in ['open', 'close']:
    os.mkdir(str(pathlib.Path(data_dir, 'train', f)))
    os.mkdir(str(pathlib.Path(data_dir, 'valid', f)))

# move random choice files in valid
for fold in rand_chose.keys():
    for pic in rand_chose[fold]:
        if str.find(str(pic), 'open') != -1:
            os.rename(pic, str(pathlib.Path(valid_dir, 'open', str.split(str(pic), '\\')[-1])))
        else:
            os.rename(pic, str(pathlib.Path(valid_dir, 'close', str.split(str(pic), '\\')[-1])))


# move the rest of files in train
for fold in fold_names:
    pics = list(fold.glob('*.jpg'))
    for pic in pics:
        if str.find(str(pic), 'open') != -1:
            os.rename(pic, str(pathlib.Path(train_dir, 'open', str.split(str(pic), '\\')[-1])))
        else:
            os.rename(pic, str(pathlib.Path(train_dir, 'close', str.split(str(pic), '\\')[-1])))

# delete the empty folders
for fold in fold_names:
    os.removedirs(fold)

count = len(list(data_dir.glob('*/*/*.jpg')))
print(f'The number of all images is {count}')


# for dr in os.listdir(valid_dir):
#     pth = pathlib.Path(valid_dir, dr)
#     print(f"Directory {dr} contains {len(list(pth.glob('*.jpg')))} images\n")


# set the class names
class_names = np.array([x.name for x in train_dir.glob('*')])
print(f'The classes are: {class_names}')


# print a few images
close_img = list(train_dir.glob('close*/*'))

fig, ax = plt.subplots(nrows=1, ncols=4)
for i, img in enumerate(close_img[:4]):
    ax[i].imshow(cv2.imread(str(img)))


open_img = list(train_dir.glob('open*/*'))

fig, ax = plt.subplots(nrows=1, ncols=4)
for i, img in enumerate(open_img[:4]):
    ax[i].imshow(cv2.imread(str(img)))


# define function to extract a batch of images
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(8):
        ax = plt.subplot(2, 4, n + 1)
        plt.imshow(image_batch[n])
        plt.title(class_names[label_batch[n] == 1][0].title())
        plt.axis('off')


# generate the input data

# extract all image image paths from all forder
list_train_ds = tf.data.Dataset.list_files(str(train_dir) + '/*/*')
list_valid_ds = tf.data.Dataset.list_files(str(valid_dir) + '/*/*')

# set model input shape and others
batch_size = 5
img_height = 24
img_width = 24
steps_per_epoch = np.ceil(count / batch_size)


# process the path and return just the folder name - open or close
def get_label(file_path):
    parts = tf.strings.split(file_path, '\\')
    return parts[-2] == class_names


# decode the image
def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [img_height, img_width])


# combine all above
def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.image.resize(img, (224, 224))
    return img, label


# set the batch
def prepare_train(ds, cache=True):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.repeat(10)
    ds = ds.batch(batch_size)
    return ds


# apply the above
train_ds = prepare_train(list_train_ds.map(process_path))
valid_ds = prepare_train(list_valid_ds.map(process_path))


# test if all images can be decode
for i in iter(list_train_ds):
    try:
        img = tf.image.decode_jpeg(tf.io.read_file(i.numpy()), channels=3)
        next
    except:
        print(f'File with problems: {i.numpy()}')

# check if the batch is ok
image_batch, label_batch = next(iter(valid_ds))
print(f'The shape of the image batch: {image_batch.shape}')
print(f'The shape of the label batch: {label_batch}')


# download the model
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3))

# test the output shape of the extractor
feature_batch = feature_extractor_layer(image_batch)
print(f'Shape at exit from extractor {feature_batch.shape}')

# set to freez the weights
feature_extractor_layer.trainable = False

# initialize the model
model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(units=2, activation='softmax')])

model.summary()

model.compile(optimizer='Adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['acc'])

history = model.fit(train_ds,
                    epochs=10,
                    validation_data=valid_ds,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=1)

model.evaluate(valid_ds, steps=steps_per_epoch)


# predict
preds = model.predict(valid_ds, steps=1)


image_batch, label_batch = next(iter(valid_ds))

plt.figure(figsize=(10, 10))
for n in range(5):
    ax = plt.subplot(2, 4, n + 1)
    plt.imshow(image_batch[n])
    plt.title(class_names[np.argmax(preds[n])].title())
    plt.axis('off')

model.save('./open_close_eye_model')


# test the saved model and make a prediction
model = tf.keras.models.load_model('./open_close_eye_model')

img_path = './dataset_B_Eye_Images/train/close/closed_eye_2576.jpg_face_1_L.jpg'

img = tf.io.read_file(img_path)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, (224, 224))
img = tf.image.convert_image_dtype(img, tf.float32)


p = model.predict(img[np.newaxis, ...])
plt.imshow(img.numpy().astype(int))
plt.title(class_names[np.argmax(p)])
