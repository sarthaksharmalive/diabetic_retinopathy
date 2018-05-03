from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import keras

# BATCH PARAMETERS
batch_size = 1

# Initial Setup | folder structure of train and test
train_path = 'train'
test_path = 'test'

#preparing batches
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['DR', 'NO_DR'], batch_size=batch_size)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['DR', 'NO_DR'], batch_size=batch_size)

# get X_train, y_train from batch
# here X_train = imgs, and y_train = labels
imgs, labels = next(train_batches)

vgg_model = keras.applications.vgg16.VGG16()
vgg_model.summary()

model = Sequential()
for layer in vgg_model.layers:
	model.add(layer)

model.layers.pop()

for layers in model.layers:
	layer.trainable = False

model.add(Dense(2, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=1392//batch_size, validation_data=test_batches, validation_steps=222//batch_size, epochs=30, verbose=2)