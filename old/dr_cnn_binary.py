# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import keras as k
import numpy
import multiprocessing
import sys
import time
import collections
from keras.callbacks import ModelCheckpoint # Implementing callbacks to save best weights per epoch in training
import h5py                                 # Best weights saved in hdf5 format


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# resolution in pixels x pixels
resolution = 128

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (resolution, resolution, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# load_flag = 0
# if load_flag == 0:
# 	# load weights
# 	classifier.load_weights("weights.best.hdf5")
# load_flag = 1

# Compiling the CNN
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Checkpoint for the model - Selecting best weights
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=4)
callbacks_list = [checkpoint]


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

try:
	pool.terminate()
except:
	pass
n_process = 12
pool = multiprocessing.Pool(processes=n_process)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   samplewise_std_normalization=True,
                                   featurewise_std_normalization=True,
                                   zca_whitening = True,
                                   width_shift_range = 0.2,
                                   vertical_flip = True,
                                   rotation_range = 180,
                                   horizontal_flip = True)

# train_datagen.fit()

test_datagen = ImageDataGenerator(rescale = 1./255)

pool.terminate()

training_set = train_datagen.flow_from_directory('train_resized',
                                                 target_size = (resolution, resolution),
                                                 batch_size = 1,
                                                 save_to_dir = 'aug_img',
                                                 save_prefix = 'aug_',
                                                 save_format = 'jpeg',
                                                 classes = ['DR', 'NO_DR'],
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_resized',
                                            target_size = (resolution, resolution),
                                            batch_size = 1,
                                            classes = ['DR', 'NO_DR'],
                                            class_mode = 'binary')

# There are 25810 images in DR, and 9316 images in NO_DR
# class weights are put accordingly
class_weight = {0:1 , 1:2.77 }

classifier.fit_generator(training_set,
                         steps_per_epoch = 28101,
                         epochs = 24,
                         callbacks = callbacks_list,
                         class_weight = class_weight,
                        # use_multiprsuocessing = True,
                         validation_data = test_set,
                         validation_steps = 7025)
# if __name__ == '__main__':
# 	Pool()