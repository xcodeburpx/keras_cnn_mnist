"""
Created on Wed Jan  4 21:04:42 2017

@author: kinshiryuu
"""

#Import libraries

from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Convolution2D, MaxPooling2D, merge, Flatten, \
                            Activation, Dropout, Dense
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, CSVLogger, ProgbarLogger, ReduceLROnPlateau
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

K.set_image_dim_ordering('th')

im_shape = (28,28,1)    # Tensorflow dedicated
im_shape_2 = (1,28,28)  # Theano dedicated

kernel = (4, 4)
pooling = (2, 2)
#Download data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#Divide dataset into train and test datasets
X_train, y_train = mnist.train.images, mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels

#Reshape data
X_train = X_train.reshape(X_train.shape[0],im_shape_2[0], im_shape_2[1], im_shape_2[2])
X_test = X_test.reshape(X_test.shape[0],im_shape_2[0], im_shape_2[1], im_shape_2[2])

# Create model

main_input = Input(shape=im_shape_2, name="main_input")

h_1 = Convolution2D(nb_filter=32, nb_row=kernel[0], nb_col=kernel[1],init="uniform" ,border_mode='valid', name="Hidden_Conv_1", input_shape=im_shape_2)(main_input)
h_1 = Activation("tanh")(h_1)
h_1 = MaxPooling2D(pool_size=pooling)(h_1)
h_1 = Dropout(0.25)(h_1)

h_1_1 = Convolution2D(nb_filter=32, nb_row=kernel[0], nb_col=kernel[1],init="lecun_uniform", border_mode='valid', name='Hidden_Branch_1')(h_1)
h_1_1 = Activation('tanh')(h_1_1)
h_1_1 = MaxPooling2D(pool_size=pooling)(h_1_1)
h_1_1 = Dropout(0.25)(h_1_1)

h_1_2 = Convolution2D(nb_filter=32, nb_row=kernel[0], nb_col=kernel[1],init="lecun_uniform", border_mode='valid', name='Hidden_Branch_2')(h_1)
h_1_2 = Activation('tanh')(h_1_2)
h_1_2 = MaxPooling2D(pool_size=pooling)(h_1_2)
h_1_2 = Dropout(0.25)(h_1_2)

h_2 = merge([h_1_1, h_1_2],mode="mul", name="Merged_Conv")
h_2 = Dropout(0.5)(h_2)

h_3 = Flatten()(h_2)
h_3 = Dense(32, init='normal')(h_3)
h_3 = Activation("tanh")(h_3)
h_3 = Dropout(0.25)(h_3)

main_output = Dense(10, activation="sigmoid", name="main_output") (h_3)

tensorboard = TensorBoard(log_dir='tensorflow_logs/examples', histogram_freq=0, write_graph=True, write_images=True)
logger = CSVLogger(filename='tensorflow_logs/logs.csv')
progbar = ReduceLROnPlateau(factor=0.08, patience=5, verbose=1)

model = Model(input=[main_input], output=[main_output])
model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=['accuracy'])

# Train the model
model.fit({'main_input': X_train},
          {'main_output': y_train},
          nb_epoch=50, batch_size=48,
          validation_data=(X_test, y_test),  callbacks=[tensorboard, logger, progbar], shuffle=True)


for layer in model.layers:
    print(layer.get_config())
    print(layer.get_weights())
    #print(layer.W.value())
    print()


plot(model, to_file="conv_1.png", show_shapes=True)
model.save('my_model_2.h5')
print("\n\nMODEL SAVED\n\n")

image = mpimg.imread("conv_1.png")
plt.imshow(image)
plt.show()


print("\nFINISHED!!!!!!\n\n")
