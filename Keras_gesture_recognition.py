#!/usr/bin/env python
from keras.layers import Input, Dense,Conv2D,MaxPooling2D,Flatten,Reshape
from keras.models import Sequential
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
#import pillow

handle=open("mixed_images_tf_ravled.pickle","rb")
data=pickle.load(handle)
handle.close()

handle_=open("mixed_names_tf_ravled.pickle","rb")
lables=pickle.load(handle_)
handle_.close()
#print(data.head())
''''a=data
test_data=a.as_matrix()
#print(test_data[0].shape)
#print(test_data.shape)
shape_=test_data.shape
print(shape_[0])
first1=shape_[0]
reshape_try=test_data.reshape(first1,1,200,200)
print(reshape_try.shape)
#break_=2
for x in a:
 c=
 b=a.flatten()
 test_data.append(b)
print(test_data)
print(test_data_.shape)'''

Data_Train,Data_test,Lables_train,Lables_test=train_test_split(data,lables,test_size=0.3,random_state=0)

#print('Training set', Data_Train, Lables_train.shape)
#print('Validation set', valid_dataset.shape, valid_labels.shape)
#print('Test set', Data_test,Lables_test.shape)
#tf_train_dataset = tf.placeholder(tf.int,shape=(None,200,200,1))
#tf_train_labels = tf.placeholder(tf.int,shape=(None,3))
#tf_test_dataset = tf.constant(Data_test_)
#tf_test_lables=tf.constant(Lables_test)
#print(Data_Train.shape)
#before_network_data=Data_Train.reshape(200,200,1)
#print(before_network_data.shape)

model = Sequential()
model.add(Reshape((200,200,1), input_shape=(40000,)))
model.add(Conv2D(6, kernel_size=(28, 28), strides=(1, 1),activation='relu'))
model.add(Conv2D(6, kernel_size=(14, 14), strides=(1, 1),activation='relu'))
model.add(Conv2D(16, kernel_size=(10, 10), strides=(1, 1),activation='relu'))
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Conv2D(64, (5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=120, activation='tanh'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(Data_Train,Lables_train,validation_split=0.3,epochs=200)
score=model.evaluate(Data_test,Lables_test)
names=model.metrics_names

print(score,"score")
print(names,"names")

model.save("keras_convmodel_200epoch")

'''
model = Sequential()
model.add(Dense(units=60, activation='tanh', input_dim=200))
model.add(Dense(units=60, activation='relu'))
#model.add(Dense(units=3, activation='softmax'))
model.add(Dense(units=60, activation='tanh'))
model.add(Dense(units=60, activation='relu'))
#model.add(Dense(units=100, activation='tanh'))
#model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=3, activation='softmax'))
#model.add(Dense(units=50, activation='relu'))
#model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(Data_Train,Lables_train,validation_split=0.3,epochs=50)
score=model.evaluate(Data_test,Lables_test)
names=model.metrics_names

print(score,"score")
print(names,"names")
'''

