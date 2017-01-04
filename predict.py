from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py
import os
import gc

weights_path = 'large_files/vgg16_weights.h5'
top_model_weights_path = 'large_files/bottleneck_fc_model.h5'
# path to data files
test_data_dir = 'data/test'
n_test_samples = 12500
# input image size
height = 150
width = 150

gc.collect()


def save_bottleneck_features(img_width, img_height, weights_path, data_dir, n_samples, filename_suffix):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load VGG16 network weights
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # ignore fully connected layers
            break
        layer = f['layer_{}'.format(k)]
        weights = [layer['param_{}'.format(param)] for param in range(layer.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('VGG16 model loaded')

    # save network output for training and validation sets to files for later use
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False
    )

    # print (generator.filenames)
    with open('data/prediction_order.csv', 'w') as f:
        for name in generator.filenames:
            f.write(name)
            f.write('\n')
    bottleneck_features = model.predict_generator(generator, n_samples)
    np.save(open('large_files/bottleneck_features_%s.npy' % filename_suffix, 'wb'), bottleneck_features)
    print('bottleneck features saved')


# save_bottleneck_features(width, height, weights_path, test_data_dir, n_test_samples, 'test')
# full connected layer added to VGG16 convolution layers
test_data = np.load(open('large_files/bottleneck_features_test.npy', 'rb'))

# to load model with previously calculated weights
'''
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
'''

model = load_model(top_model_weights_path)

y_test = model.predict(test_data, batch_size=32)

with open('data/prediction_order.csv') as f:
    file_names = f.read().splitlines()

s_numbers = [int(name.split('/')[1].split('.')[0]) for name in file_names]

with open('data/predictions.csv', 'w') as f:
    for s_number, result in sorted(zip(s_numbers, y_test)):
        # p = 1 - result
        f.write(str(s_number) + ',' + '%.2f' % result)
        f.write('\n')


gc.collect()
