"""
The dataset is taken from this address: https://www.kaggle.com/jessicali9530/celeba-dataset
Because the dataset is very large, only the first around 2500 images were used.
"""

import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split


# CLASSES FOR INPUT READING
# classes for reading the image and the eye position
# Reader_TF is made to be used with tf.data.Dataset.from_generator. not used
class Reader():
    def __init__(self, img_folder, data_file, batchsize=5):
        self.img_folder = pathlib.Path(img_folder)
        self.img_list = list(self.img_folder.glob('./*.jpg'))
        self.data_file = data_file
        self.locations = np.genfromtxt(self.data_file, skip_header=True, delimiter=',').astype(int)
        self.batchsize = batchsize

    def __len__(self):
        self.n_batch = (len(self.img_list) // self.batchsize) - 1
        self.n_last_batch = len(self.img_list) % self.batchsize
        self.n_batch = self.n_batch if self.n_last_batch == 0 else self.n_batch + 1
        return self.n_batch

    def __getitem__(self, i):
        batch = self.batchsize if i + 1 < len(self) else self.n_last_batch
        start = i * self.batchsize
        stop = start + batch

        img_short_list = list()

        for im in self.img_list[start:stop]:
            img_short_list.append(cv2.imread(str(im)) / 255.)
        img_batch = np.stack(img_short_list, axis=0)

        label_batch = self.locations[start: stop, 1:5]

        return img_batch, label_batch

class Reader_TF(Reader):

    def __init__(self, img_folder, data_file, batchsize=5, shape=None):
        super().__init__(img_folder, data_file, batchsize)
        self.shape = shape

    def __getitem__(self, item):
        batch = self.batchsize if item + 1 < len(self) else self.n_last_batch
        start = item * self.batchsize
        stop = start + batch

        img_short_list = list()
        label_short_list = list()

        for im, lab in zip(self.img_list[start:stop], self.locations[start: stop, 1:5]):
            img, w, h = self.resize_image(im)
            label = self.scale_label(lab, w, h)

            img_short_list.append(img)
            label_short_list.append(label)

        img_batch = np.stack(img_short_list, axis=0)

        label_batch = np.stack(label_short_list, axis=0)

        return img_batch, label_batch

    def resize_image(self, path):
        img = tf.io.read_file(str(path))
        img = tf.image.decode_jpeg(img)
        w, h = img.shape[:2]
        if self.shape:
            img = tf.image.resize(img, size=self.shape)
            img = img / 255.
        return img, w, h

    def scale_label(self, label, w, h):
        x_s = (min(label[1], label[3]) - 20) / h
        x_f = (max(label[1], label[3]) + 20) / h
        y_s = (min(label[0], label[2]) - 20) / w
        y_f = (max(label[0], label[2]) + 20) / w
        return np.array([x_s, x_f, y_s, y_f])

# get some examples an plot them
generator = Reader('./img_face', './list_landmarks_align_celeba.csv')
x, y = generator[487]

# plot
fig, ax = plt.subplots(1, 5)
i = 0
for img, box in zip(x, y):
    a = min(box[1], box[3]) - 20
    b = max(box[1], box[3]) + 20
    c = min(box[0], box[2]) - 20
    d = max(box[0], box[2]) + 20

    img[:, :, 0][a:b, c:d] = 1

    ax[i].imshow(img)
    i+=1

# # make the generator a function, as neede for TF
# generator_tf = Reader_TF('./img_face', './list_landmarks_align_celeba.csv', shape=(224, 224))
#
# def gen():
#     return generator_tf
#
# ds = tf.data.Dataset.from_generator(gen,
#                                     (tf.float32, tf.float32),
#                                     (tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None, 4])))




# PREPARE DATA ----------
img_folder = pathlib.Path('./img_face')
img_list = list(img_folder.glob('./*.jpg'))
img_list = [str(img) for img in img_list]
locations = np.genfromtxt('./list_landmarks_align_celeba.csv', skip_header=True, delimiter=',').astype(int)
locations = locations[:len(img_list), 1:5]

# SPLIT ----------
x_train, x_test, y_train, y_test = train_test_split(img_list, locations, test_size=0.1)

# CREATE TF DATASETS ----------
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# function to read, decode, resize images / expand eye point to a box
# will be mapped onto the datasets
@tf.function
def prep_data(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    k = tf.shape(img)
    img = tf.image.resize(img, size=(224, 224))
    img = img / 255.

    x_s = tf.divide(tf.add(tf.minimum(label[1], label[3]), -20), k[0])
    x_f = tf.divide(tf.add(tf.maximum(label[1], label[3]), 20), k[1])
    y_s = tf.divide(tf.add(tf.minimum(label[0], label[2]), -20), k[0])
    y_f = tf.divide(tf.add(tf.maximum(label[0], label[2]), 20), k[1])
    label = [x_s, x_f, y_s, y_f]
    return img, label

train_ds = train_ds.map(prep_data)
train_ds = train_ds.batch(20).shuffle(20).repeat(5)

test_ds = test_ds.map(prep_data)
test_ds = test_ds.batch(5)


# DOWNLOAD A FEATURE EXTRACTOR ---------
mobile_net = hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4', input_shape=(224, 224, 3))
mobile_net.trainable = False

# CREATE THE MODEL ----------
model = tf.keras.Sequential(mobile_net)
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mse',
              metrics=['Accuracy'])

# TRAIN AND EVALUATE ----------
history = model.fit(train_ds, epochs=5)

loss = history.history['loss']
fig, ax = plt.subplots(1, 1)
ax.plot(loss)
ax.set_title('Model Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_xlim(0, 5)
ax.set_xticks(range(5))


# MAKE PREDICTIONS ---------
y_pred = model.predict(test_ds)

# because the input was scaled, it needs the reverse process
def unscale_label(label, w, h):
    x_s = label[0] * w
    x_f = label[1] * w
    y_s = label[2] * h
    y_f = label[3] * h
    return np.array([x_s, x_f, y_s, y_f], dtype='int64')


new_label = list()

for l in y_pred:
    new_label.append(unscale_label(l, 178, 218))

new_label = np.stack(new_label, axis=0)


fig, ax = plt.subplots(1, 5)
i = 0
for img, box, new_box in zip(x_test[:5], y_test[:5], new_label[:5]):

    img = cv2.imread(img)

    a = min(box[1], box[3]) - 20
    b = max(box[1], box[3]) + 20
    c = min(box[0], box[2]) - 20
    d = max(box[0], box[2]) + 20

    aa = new_box[2]
    bb = new_box[3]
    cc = new_box[0]
    dd = new_box[1]

    img[:, :, 0][a:b, c:d] = 255

    img[:, :, 1][cc:dd, aa:bb] = 255

    ax[i].imshow(img)
    i+=1

model.save('eye_box')