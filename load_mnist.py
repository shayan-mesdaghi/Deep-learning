# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:18:30 2018

@author: FaraDars
"""
from keras.datasets import mnist


# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data attributes
print("train_images dimentions: ", train_images.ndim)
print("train_images shape: ", train_images.shape)
print("train_images type: ", train_images.dtype)

X_train = train_images.reshape(60000, 784)
X_test = test_images.reshape(10000, 784)

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

from keras.utils import np_utils
Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)   

#==================================================
# Creating our model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt

myModel = Sequential()
myModel.add(Dense(500, activation='relu', input_shape=(784,)))
myModel.add(Dropout(20))
myModel.add(Dense(100, activation='relu'))
myModel.add(Dropout(20))
myModel.add(Dense(10, activation='softmax'))

myModel.summary()
#myModel.compile(optimizer=SGD(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])
myModel.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

#==================================================
# Train our model
hist = myModel.fit(X_train, Y_train, batch_size=128, epochs=10, validation_split=0.2)


# Evaluation
test_loss, test_acc = myModel.evaluate(X_test, Y_test)
test_labels_p = myModel.predict(X_test)
import numpy as np
test_labels_p = np.argmax(test_labels_p, axis=1)


history = hist.history

losses = history['loss']
val_losses = history['val_loss']
accuracies = history['accuracy']
val_accuracies = history['val_accuracy']

plt.figure()  
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses, 'red')
plt.grid()
plt.plot(val_losses, 'blue')
plt.legend(['loss', 'val_loss'])
    
plt.figure()
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.grid()
plt.plot(accuracies)
plt.plot(val_accuracies)
plt.legend(['accuracy', 'val_accuracy'])