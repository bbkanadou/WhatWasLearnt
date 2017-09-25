import numpy as np
import cv2
import glob
from random import shuffle
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
from keras.models import Sequential
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

class Ones(Layer):

    def __init__(self, **kwargs):
        self.output_dim = 1
        super(Ones, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
	print 'nonmaisallo{}'.format(input_shape)
        self.output_dim = (input_shape[1],)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=((input_shape[1],)),
                                      initializer='Ones',
                                      trainable=False)
        super(Ones, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x*self.kernel

    def trainable(self, talebool):
	self.trainable=talebool

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
