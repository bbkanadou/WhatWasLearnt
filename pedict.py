import numpy as np
import cv2
import glob
from random import shuffle
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
#Convolution2D, MaxPooling2D, 
from keras.optimizers import Adam, SGD
from keras import regularizers
import h5py
from keras import backend as K
from keras.engine.topology import Layer

class Proutlayer(Layer):

    def __init__(self, **kwargs):
        self.output_dim = 1
        super(Proutlayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.output_dim = (50*50*3,)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=((50*50*3,)),
                                      initializer='Ones',
                                      trainable=True)
        super(Proutlayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x*self.kernel

    def trainable(self, talebool):
	self.trainable=talebool

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


height_input = 50 #height of the input image of the NN
width_input = 50  #width  of the input image of the NN
color = True   #True means images will be used as bgr, False as grayscale
folder_name = 'data_set'
train_ratio = 1 #Percentage of pictures used for train set
val_split = 0.2
if color == True:
	depth = 3
else:
	depth = 1

def load_image(height, width, picture, color):
	print picture
	if color == True:
		img = cv2.imread(picture)
		print img.shape
		img = cv2.resize(img,(height,width))
	else:
		img = cv2.imread(picture,0)
		img = cv2.resize(img,(height,width))
		img = np.expand_dims(img, axis = -1)
	#cv2.imwrite('0.png',img)
	return img

# returns a compiled model
# identical to the previous one
classes = np.load("variable_names.npy")

model = load_model('my_model.h5',{"Proutlayer":Proutlayer})


image = load_image(height_input, width_input, "0.png", color)
image = np.expand_dims(image, axis=0)
image = image / 255.
prediction = model.predict_on_batch(image)
prediction = prediction.argmax()
print classes[prediction]


for l in model.layers:
	l.trainable = False
model.layers[0].trainbale = True

image2 = np.ones((1,50,50,3))
y2 = np.zeros((1,4))
y2[0,1]= 1

#
model.layers[1].trainable = True
for l in model.layers:
	print l.trainable

sgd = SGD(lr=0.001)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
model.fit(image2, y2, epochs=1000)
model.save('my_model2.h5')

