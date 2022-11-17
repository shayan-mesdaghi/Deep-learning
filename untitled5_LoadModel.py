# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:16:27 2022

@author: SHM
"""

from keras.models import load_model 

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
import datetime
#1
myModel_Loaded = load_model('model.h5')

#2 Compile model
myModel_Loaded.compile(optimizer=SGD(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])
myModel_Loaded.summary()

#3 load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
X_train = train_images.reshape(60000, 784)
X_test = test_images.reshape(10000, 784)

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255


Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)   

#4
#===============================================================================
# Predict + Visualization
start = datetime.datetime.now()
test_labels_p = myModel_Loaded.predict(X_test)
test_labels_P = np.argmax(test_labels_p, axis=1)
end = datetime.datetime.now()
elapsed = end-start
print('Total training time: ', str(elapsed))