from keras.models import Sequential, load_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy
import keras
from keras import regularizers
import numpy as np
import tensorflow as tf
import random as rn

# Fixing Random SEED for reproducible results
import os
# os.environ['PYTHONHASHSEED']= '666'
# np.random.seed(8)
# rn.seed(314)
# tf.set_random_seed(5)
# BATCH PARAMETERS
batch_size = 32
flag_m = 0

# Initial Setup | folder structure of train and test
train_path = 'train'
valid_path = 'validation'
test_path = 'test'
#preparing batches
train_batches = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,rotation_range=10).flow_from_directory(train_path, target_size=(224,224), classes=['0', '4'], batch_size=batch_size)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['0', '4'], batch_size=batch_size)

# get X_train, y_train from batch
# here X_train = imgs, and y_train = labels
imgs, labels = next(train_batches)

base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3))
# base_model.summary()

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
x = Dropout(0.7)(x)
# and a logistic layer -- let's say we have 2 classes
predictions = Dense(2, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)


# model = Sequential()
# for layer in vgg_model.layers:
# 	model.add(layer)

# model.layers.pop()
# for layer in base_model.layers:
# 	layer.trainable = True

for layer in base_model.layers:
	layer.trainable = False


# model.add(Dense(2, activation='softmax'))
# if flag_m==0:
# model = load_model('bw_imagenet_3class_nobucket_t1.hdf5')
# 	model.summary()
# 	flag_m=flag_m+1

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]: #249
#    layer.trainable = False
# for layer in model.layers[249:]: #249
#    layer.trainable = True

model.compile(optimizer=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy',categorical_accuracy])

# Callback that streams epoch results to a csv file.
logger = keras.callbacks.CSVLogger('inceptionv3_train_bicross_1_withdrpt.log', separator=',', append=True)

# Preparing checkpoint
checkpointer = ModelCheckpoint(filepath='bicross_inceptionv3_try1_withdrpt.hdf5', verbose=1, save_best_only=True, mode='max', monitor='val_acc')

model.fit_generator(train_batches, shuffle=True, steps_per_epoch=988//batch_size, validation_data=valid_batches, validation_steps=220//batch_size, epochs=250, callbacks=[checkpointer, logger])