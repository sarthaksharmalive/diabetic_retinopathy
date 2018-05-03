from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import keras

# BATCH PARAMETERS
batch_size = 4
flag_m = 0

# Initial Setup | folder structure of train and test
train_path = 'train'
test_path = 'test'

#preparing batches
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['DR', 'NO_DR'], batch_size=batch_size)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['DR', 'NO_DR'], batch_size=batch_size)

# get X_train, y_train from batch
# here X_train = imgs, and y_train = labels
imgs, labels = next(train_batches)

base_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
# base_model.summary()

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

for layer in base_model.layers:
	layer.trainable = False

# model.add(Dense(2, activation='softmax'))
if flag_m==0:
	model = load_model('best_weights_resnet50_hyp1.hdf5')
	model.summary()
	flag_m=flag_m+1





model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
, loss='categorical_crossentropy', metrics=['accuracy'])

# Callback that streams epoch results to a csv file.
logger = keras.callbacks.CSVLogger('resnet50_train_2.log', separator=',', append=True)

# Preparing checkpoint
checkpointer = ModelCheckpoint(filepath='best_weights_resnet50_try2.hdf5', verbose=1, save_best_only=True, mode='max', monitor='val_acc')

model.fit_generator(train_batches, steps_per_epoch=6184//batch_size, validation_data=test_batches, validation_steps=664//batch_size, epochs=120, callbacks=[checkpointer, logger])
print(flag_m)