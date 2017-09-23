import numpy as np
import keras
import h5py
from keras.models import load_model, Sequential
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


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
model = Sequential()
print "wut"
model = load_model('my_model.h5')


image = load_image(height_input, width_input, "0.png", color)
image = np.expand_dims(image, axis=0)
prediction = model.predict_on_batch(image)
prediction = prediction.argmax()
print classes[prediction]

print type(model.layers)
new_model = Sequential()
new_model.add(LocallyConnected1D(64, 3, input_shape=(50, 50, 3)))
new_model.add(model.layer[0])
model.trainable = False
print model.trainable
