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
from sklearn.metrics import roc_auc_score, roc_curve, auc
import itertools
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# Fixing Random SEED for reproducible results
os.environ['PYTHONHASHSEED']= '666'
np.random.seed(8)
#rn.seed(314)
tf.set_random_seed(5)


conf_plot_name='Conf_bicross_resnet50_try5_withdrpt_aug.png'
roc_plot_name='ROC_bicross_resnet50_try5_withdrpt_aug.png'
load_model_name='bicross_resnet_model.hdf5'

batch_size = 1

train_path = 'train'
valid_path = 'validation'
test_path = 'test'


test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['0', '4'], batch_size=batch_size)

# Load model
model = load_model(load_model_name)
model.summary()
ylab=np.array([])
x=0

# Takes 100 images in test folder one by one and creates their true label in ylab (y lables)
while x<100*batch_size:
	test_imgs, test_labels = next(test_batches)
	test_labels = test_labels[:,0]
	ylab=np.concatenate((ylab,test_labels), axis=0)
	x+=1

# converting true lables to numpy array
ylab = list(ylab)
ylab = np.float32(ylab)
print(test_labels)

# Predicting on test images with our model and get lables in numpy array
predictions = model.predict_generator(test_batches, steps=100, verbose=0)
predictions = predictions[:,0]
predictions = np.where(predictions>0.5, 1, 0)
predictions = np.float32(predictions)


# outputs Confusion matrix for our test samples
cm = confusion_matrix(ylab, predictions)
print(cm)

# defining ROC function
def plot_roc(pred,y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(roc_plot_name)

# Defining confusion matrix function
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
    plt.savefig(conf_plot_name)


cm_plot_labels = ['cat_0','cat_4']

# Plots confusion matrix
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
# Plots ROC
plot_roc(ylab,predictions)
