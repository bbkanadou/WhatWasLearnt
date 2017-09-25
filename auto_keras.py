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

class Proutlayer(Layer):

    def __init__(self, **kwargs):
        self.output_dim = 1
        super(Proutlayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
	print "ok1"
        self.output_dim = (50*50*3,)
	print "ok1bis"
        self.kernel = self.add_weight(name='kernel', 
                                      shape=((50*50*3,)),
                                      initializer='Ones',
                                      trainable=False)
	print "ok2"
	print self.kernel
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

def load_sets():
	true == false
	x_set = np.load('x_set.npy')
	y_set = np.load('y_set.npy')
	classes = np.load('variable_names.npy')
	print 'classes : {}'.format(classes)
	print ('sets loaded')
	return x_set, y_set

def count_examples(folder_names):
	tot_pictures = 0
	for folder in folder_names:
		pictures = folder+'/*'
		for picture in glob.glob(pictures):
			tot_pictures+=1
	return tot_pictures

def create_variable_names(folder_names, name_size):
	classes = []
	for folder in folder_names:
		classes.append(folder[name_size+1:])
	np.save('variable_names.npy',classes)
	print ('classes saved')
	return classes

def load_image(height, width, picture, color):
	if color == True:
		img = cv2.imread(picture)
		img = cv2.resize(img,(height,width))
	else:
		img = cv2.imread(picture,0)
		img = cv2.resize(img,(height,width))
		img = np.expand_dims(img, axis = -1)
	cv2.imwrite('0.png',img)
	return img

def create_sets(folder_name, height, width, color):
	folder_names = glob.glob(folder_name+'/*')
	name_size = len(folder_name)
	classes = create_variable_names(folder_names, name_size)
	print 'classes : {}'.format(classes)
	tot_pictures = count_examples(folder_names)
	
	y_set = np.zeros((tot_pictures,len(classes)))
	if color == True:
		x_set = np.zeros((tot_pictures, height, width, 3))
	elif color == False:
		x_set = np.zeros((tot_pictures, height, width, 1))
	else:
		print 'color must be True or False'

	class_nb = 0
	tot = 0
	for i in range(len(folder_names)):
		pictures = folder_names[i]+'/*'
		for picture in glob.glob(pictures):
			x_set[tot,:,:,:] = load_image(height, width, picture,color)
			y_set[tot,class_nb] = 1
			tot +=1
		class_nb +=1
	
	np.save('x_set.npy',x_set)
	np.save('y_set.npy',y_set)
	print ('sets saved')	
	return x_set, y_set

def create_train_test(x_set, y_set, train_ratio):
	nb_examples = y_set.shape[0]
	train_nb = int(train_ratio*nb_examples)
	test_nb = nb_examples - train_nb
	x_train_shape = (train_nb,)+x_set.shape[1:]
	x_test_shape  = (test_nb,) +x_set.shape[1:]
	y_train_shape = (train_nb,)+y_set.shape[1:]
	y_test_shape  = (test_nb,) +y_set.shape[1:]

	random_indices = range(nb_examples)
	shuffle(random_indices)

	x_train = np.zeros(x_train_shape)
	x_test = np.zeros(x_test_shape)
	y_train = np.zeros(y_train_shape)
	y_test = np.zeros(y_test_shape)

	for cpt in range(len(random_indices)):
		if cpt < train_nb:
			x_train[cpt,:,:,:] = x_set[random_indices[cpt],:,:,:]
			y_train[cpt,:]     = y_set[random_indices[cpt],:]
		else:
			x_test[cpt-train_nb,:,:,:] = x_set[random_indices[cpt-train_nb],:,:,:]
			y_test[cpt-train_nb,:]     = y_set[random_indices[cpt-train_nb],:]
	return x_train, x_test, y_train, y_test			

try:
	assert 1==2
	print "nope"
	x_set,y_set = load_sets()
except:
	print "yep"
	x_set, y_set = create_sets(folder_name, height_input, width_input, color)
	

X_train, test_set, Y_train, y_test = create_train_test(x_set, y_set, train_ratio)
print test_set.shape, y_test.shape

X_train, test_set = X_train/255., test_set/255.

print X_train.shape
print Y_train.shape
model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
#model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(height_input, width_input,depth)))
model.add(Flatten(input_shape=(height_input, width_input,depth)))
model.add(Proutlayer(input_shape=(height_input, width_input,depth)))
model.add(Activation('relu'))
model.add(Reshape((height_input, width_input,depth)))

#input_shape=(height_input, width_input,depth), data_format="channels_last",
model.add(Conv2D(16, (3,3),strides=(1, 1), padding='valid',  activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (1,1),strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

# Note: Keras does automatic shape inference.
model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01),use_bias=True))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01),use_bias=True))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(Y_train.shape[1]))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
sgd = SGD(lr=0.001)#-1000*0.0000001, decay=0.0000001)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

for i in range(10):
	model.fit(X_train, Y_train, validation_split = val_split, batch_size=32, epochs=1000)
	model.save('my_model_sgd{}.h5'.format(i))
