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

base_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
# base_model.summary()

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l1(0.000000005))(x)
x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have 2 classes
predictions = Dense(2, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)


# model = Sequential()
# for layer in vgg_model.layers:
# 	model.add(layer)

# model.layers.pop()

for layer in base_model.layers:
	layer.trainable = False

model = load_model('bicross_resnet50_try2_withdrpt.hdf5')

model.compile(optimizer=keras.optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
, loss='binary_crossentropy', metrics=['accuracy'])

# Callback that streams epoch results to a csv file.
logger = keras.callbacks.CSVLogger('resnet50_train_bicross_5_withdrpt_aug.log', separator=',', append=True)

# Preparing checkpoint
checkpointer = ModelCheckpoint(filepath='bicross_resnet50_try5_withdrpt_aug.hdf5', verbose=1, save_best_only=True, mode='max', monitor='val_acc')

model.fit_generator(train_batches, steps_per_epoch=1388//batch_size, validation_data=valid_batches, validation_steps=220//batch_size, epochs=500, callbacks=[checkpointer, logger])