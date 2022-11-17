# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:58:36 2022

@author: SHM
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils

import datetime#1
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


Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)   

#==================================================
# Creating our model


myModel = Sequential()
myModel.add(Dense(500, activation='relu', input_shape=(784,)))
myModel.add(Dropout(20))
myModel.add(Dense(100, activation='relu'))
myModel.add(Dropout(20))
myModel.add(Dense(10, activation='softmax'))

myModel.summary()
myModel.compile(optimizer=SGD(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])

# Plot model #3
#1. install --> conda install pydot
#2. install --> conda install graphviz
# if dont setup please use pip install ...
#3. after install you should add graphviz to path enviroment windows
#4. copy path 'C:\ProgramData\Anaconda3\pkgs\graphviz-2.38-hfd603c8_2\Library\bin\graphviz'
#5. open enviroment windows open enviroment variables & in system variables open path and then new past path step 4 and ok then restart your PC 
from keras.utils import plot_model
plot_model(myModel, to_file='Dense_model.pdf', show_shapes=True)#pdf png
#==================================================
# Train our model
start = datetime.datetime.now()
hist = myModel.fit(X_train, Y_train, batch_size=128, epochs=3, validation_split=0.2)
end = datetime.datetime.now()
elapsed = end-start
print('Total training time: ', str(elapsed))
# Evaluation
test_loss, test_acc = myModel.evaluate(X_test, Y_test)
test_labels_p = myModel.predict(X_test)
test_labels_p = np.argmax(test_labels_p, axis=1)

print("Test_Accuracy: {:.2f}%".format(myModel.evaluate(np.array(X_test), np.array(Y_test))[1]*100))#2

#==================================================
#Visual
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