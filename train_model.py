import argparse
import tensorflow as tf
import tensorflow_hub as hub


parser = argparse.ArgumentParser(description='''Train model over eys or other images. should be used in conjunction
                                             with split_train_test, by not necessary''')
parser.add_argument('--data_dir', type=str, help='Where to find images', required=True)
parser.add_argument('--train_dir', type=str, help='Should be under dir_name')
parser.add_argument('--valid_dir', type=str, help='Mandatory if train_dir exists')
parser.add_argument('--labels', type=str,
                    nargs='*', help='''Folder names that designate the eye label. Same structure in train and valid''',
                    required=True)
parser.add_argument('--model_name', help='The name of the model save', required=True)


args = parser.parse_args()
data_dir = str(args.data_dir)
train_dir = str(args.train_dir)
valid_dir = str(args.valid_dir)
class_names = args.labels
model_name = str(args.model_name)


if train_dir is not None:
    if valid_dir is None:
        raise Exception('If train_dir is provided, valid_dir must be provided')
    else:
        # extract all image image paths from all forder
        list_train_ds = tf.data.Dataset.list_files(str(data_dir) + '/' + str(train_dir) + '/*/*')
        list_valid_ds = tf.data.Dataset.list_files(str(data_dir) + '/' + str(valid_dir) + '/*/*')
else:
    # extract all image image paths from all forder
    list_train_ds = tf.data.Dataset.list_files(str(data_dir) + '/*')

# set model input shape and others
batch_size = 5
img_height = 24
img_width = 24
steps_per_epoch = 500


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
if valid_dir is not None:
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

print(model.summary())

model.compile(optimizer='Adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['acc'])

callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs' + model_name,
                                            update_freq='epoch',
                                            histogram_freq=1)]

history = model.fit(train_ds,
                    epochs=10,
                    validation_data=valid_ds,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=callbacks)

model.evaluate(valid_ds, steps=steps_per_epoch)

model.save('./' + model_name)

