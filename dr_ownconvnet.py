from keras.models import Sequential, load_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, LeakyReLU
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
batch_size = 8
flag_m = 0
resolution = 224
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

model = Sequential()

# Input Layer
model.add(Conv2D(32, kernel_size=(7,7), strides=(1,1), input_shape=(resolution,resolution,3), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.3))
# Maxpool layer 1
model.add(MaxPooling2D(pool_size = (5,5), strides=(1, 1)))
#model.add(Dropout(0.2))

# Layer 2
model.add(Conv2D(64, kernel_size=(5,5), strides=(2,2), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.3))
# Maxpool layer 2
model.add(MaxPooling2D(pool_size = (3, 3), strides=(2, 2)))
#model.add(Dropout(0.2))

# Layer 3
model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.3))
# Maxpool layer 3
model.add(MaxPooling2D(pool_size = (3, 3), strides=(2, 2)))
#model.add(Dropout(0.2))

# Layer 4
model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.3))
# Maxpool layer 4
model.add(MaxPooling2D(pool_size = (3, 3), strides=(2, 2)))
#model.add(Dropout(0.2))

# Layer 5
model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.5))

# Flattening
model.add(Flatten())

# Dense Layers
model.add(Dense(output_dim=512, kernel_regularizer=regularizers.l2(0.1)))
model.add(Dropout(0.4))

model.add(Dense(output_dim=2, activation='sigmoid'))

model.summary()

# Load Model
model = load_model('bicross_own_convnet_1.hdf5')
# Compile
model.compile(optimizer=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='mean_squared_error', metrics=['accuracy'])

# Callback that streams epoch results to a csv file.
logger = keras.callbacks.CSVLogger('ms_own_convnet_1.log', separator=',', append=True)

# earlystop = keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50, mode='auto')

checkpointer = ModelCheckpoint(filepath='ms_own_convnet_1.hdf5', verbose=1, save_best_only=True, mode='max', monitor='val_acc')

model.fit_generator(train_batches, steps_per_epoch=988//batch_size, validation_data=valid_batches, validation_steps=220//batch_size, epochs=50, callbacks=[checkpointer, logger])