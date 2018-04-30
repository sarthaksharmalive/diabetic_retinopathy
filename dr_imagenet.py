from keras.models import Sequential, load_model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np
import tensorflow as tf
import random as rn

# Fixing Random SEED for reproducible results
import os
os.environ['PYTHONHASHSEED']= '8'
np.random.seed(666)
rn.seed(27)
tf.set_random_seed(86)
from keras import backend as k
# Forcing tensorflow to use single thread
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# k.set_session(sess)
# BATCH PARAMETERS
batch_size = 32
flag_m = 0

# Initial Setup | folder structure of train and test
train_path = 'train_resized'
test_path = 'test_resized'

#preparing batches
train_batches = ImageDataGenerator(rotation_range=60, horizontal_flip=True, vertical_flip=True, zoom_range=0.1, width_shift_range=0.1).flow_from_directory(train_path, target_size=(224,224), classes=['DR', 'NO_DR'], batch_size=batch_size)
test_batches = ImageDataGenerator(rotation_range=60, horizontal_flip=True, zoom_range=0.1, width_shift_range=0.1, vertical_flip=True).flow_from_directory(test_path, target_size=(224,224), classes=['DR', 'NO_DR'], batch_size=batch_size)


# get X_train, y_train from batch
# here X_train = imgs, and y_train = labels
imgs, labels = next(train_batches)

class_weight = {1:6.9, 0:1}

base_model = InceptionV3(include_top=False, weights='imagenet')
base_model.summary()

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

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
# 	model = load_model('best_weights_resnet50_hyp1.hdf5')
# 	model.summary()
# 	flag_m=flag_m+1

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:140]: #249
   layer.trainable = False
for layer in model.layers[140:]: #249
   layer.trainable = True

model.compile(optimizer=keras.optimizers.Adagrad(lr=0.0001, epsilon=None, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])

# Callback that streams epoch results to a csv file.
logger = keras.callbacks.CSVLogger('./logs_and_models/imagenet_train_4_alllayers_trainable_fulldata_class_balanced_10to1.log', separator=',', append=True)

earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=80, mode='auto')

checkpointer = ModelCheckpoint(filepath='./logs_and_models/best_weights_imagenet_try4_alllay_trainable_fulldata_class_balanced10to1.hdf5', verbose=1, save_best_only=True, mode='max', monitor='val_acc')

model.fit_generator(train_batches, steps_per_epoch=27448//batch_size, validation_data=test_batches, validation_steps=7025//batch_size, epochs=100, callbacks=[checkpointer, logger, earlystop], class_weight=class_weight)

