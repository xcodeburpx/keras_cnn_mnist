#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:18:17 2017

@author: kinshiryuu
"""

from tensorflow.examples.tutorials.mnist import input_data
from keras import backend as K
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import matplotlib
from keras.utils import np_utils
import numpy as np

K.set_image_dim_ordering('th')
im_shape_2 = (1,28,28)
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#Divide dataset into train and test datasets
X_train, y_train = mnist.train.images, mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels

#Reshape data
X_train = X_train.reshape(X_train.shape[0],im_shape_2[0], im_shape_2[1], im_shape_2[2])
X_test = X_test.reshape(X_test.shape[0],im_shape_2[0], im_shape_2[1], im_shape_2[2])

model = load_model("my_model_2.h5")

def plot_filters(layer, x, y):
    
    filters = layer.get_weights()[0]
    fig = plt.figure()
    plt.title("First convolutional layer weights")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    for j in range(len(filters)):
        ax = fig.add_subplot(y, x, j+1)
        im = ax.matshow(filters[j][0], cmap="Greys")
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1, 0.07, 0.05 ,0.821])
    fig.colorbar(im, cax = cbar_ax)
    plt.tight_layout()
    return plt

input_image = X_train[2:3,:,:,:]
plot_filters(model.layers[1], 8, 4)
plt.figure()
plt.imshow(input_image[0,0,:,:], cmap="Greys")

output_layer = model.layers[1].output
input_layer = model.layers[0].input
output_fn = K.function([input_layer], [output_layer])

output_image = output_fn([input_image])[0]
print("Output image shape before roll:",output_image.shape)


output_image = np.rollaxis(np.rollaxis(output_image, 3, 1), 3, 1)
print("Output image shape after roll:",output_image.shape)

fig = plt.figure()
plt.title("First convolutional layer view of output")
plt.xticks(np.array([]))
plt.yticks(np.array([]))
for i in range(32):
    ax = fig.add_subplot(4,8,i+1)
    im = ax.imshow(output_image[0,:,:,i], cmap="Greys")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([1, 0.07, 0.05 ,0.821])
fig.colorbar(im, cax = cbar_ax)
plt.tight_layout()
    
plt.show()
