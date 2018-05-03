import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# Fixing Random SEED for reproducible results
os.environ['PYTHONHASHSEED']= '666'
np.random.seed(8)
#rn.seed(314)
tf.set_random_seed(5)


batch_size = 50

train_path = 'train'
valid_path = 'validation'
test_path = 'test'


test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['0', '4'], batch_size=batch_size)

# Load model
model = load_model('bicross_resnet50_try4_withdrpt_aug.hdf5')
model.summary()

test_imgs, test_labels = next(test_batches)
ti2, te2 = next(test_batches)
test_labels = test_labels[:,0]
te2 = te2[:,0]
l = list(test_labels)+list(te2)
l = np.float32(l)
print(test_labels)

predictions = model.predict_generator(test_batches, steps=2, verbose=0)
print(predictions)
predictions = predictions[:,0]
print(predictions)
predictions = np.where(predictions>0.5, 1, 0)
predictions = np.float32(predictions)
cm = confusion_matrix(l, predictions)

# Directly taken from sklearn.metrics
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('conf_mat_bicross_resnet50_try4_withdrptbicross_resnet50_try4_withdrpt.png')


cm_plot_labels = ['cat_0','cat_4']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)