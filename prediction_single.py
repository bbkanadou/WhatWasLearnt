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
import ones_layer

model_to_load = 'my_model.h5'
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
h=3
model = load_model(mode_to_load,{"Ones":ones_layer.Ones})

"""
image = load_image(height_input, width_input, "0.png", color)
image = np.expand_dims(image, axis=0)
image = image / 255.
prediction = model.predict_on_batch(image)
prediction = prediction.argmax()
print classes[prediction]
"""

for l in model.layers:
	l.trainable = False
model.layers[0].trainbale = True

image2 = np.ones((1,50,50,3))
y2 = np.zeros((1,4))
y2[0,1]= 1

#
print len(model.layers)
model.layers[1].trainable = True
for l in model.layers:
	print l.name

nb_epochs = np.power(10,(i+1))
sgd = SGD(lr=0.0001, decay=0.0)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)


if h==0:
	optimizer_ch = adam
	loss = 'categorical_crossentropy'
	folder = 'adam_cat'
elif h==1:
	optimizer_ch = adam
	loss = 'mean_squared_error'
	folder = 'adam_mean'
elif h==2:
	optimizer_ch = sgd
	loss = 'categorical_crossentropy'
	folder = 'sgd_cat'
elif h==3:
	optimizer_ch = sgd
	loss = 'mean_squared_error'
	folder = 'sgd_mean'

model.compile(loss=loss, optimizer=optimizer_ch, metrics=['accuracy'])
model.fit(image2, y2, epochs=nb_epochs)

get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[3].output])
layer_output = get_3rd_layer_output([image2])[0]
layer_output = np.uint8(layer_output[0]*255)
layer_output = layer_output[:,:,::-1]
#cv2.imwrite('{}/1000l{}p/{}1.png'.format(folder,nb_epochs,j),layer_output)
cv2.imwrite('sgdl/1000l{}p/{}1.png'.format(nb_epochs,j),layer_output)


