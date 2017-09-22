import numpy as np
import tensorflow as tf
import cv2
import os
import glob
import keras

AMOUNT = 200
HEIGHT = 100
WIDTH  = 100
COLOR  = True

def make_folder(name):
	try:
		os.system("mkdir {}".format(name))
	except:
		pass

def det_shift(nb):
	if nb == 0:
		shift = [0,0]
	elif nb == 1:
		shift = [0,1]
	elif nb == 2:
		shift = [1,0]
	else:
		shift = [1,1]
	return shift

folders = ["generated_images", "data_set"]
sub_folders = [ "1st_corner", "2nd_corner", "3rd_corner", "4th_corner", "no_corner"]

for folder in folders:
	make_folder(folder)
for sub_folder in sub_folders:
	make_folder(folders[1]+"/"+sub_folder)
	
if COLOR:
	depth = 3
else:
	depth = 1
"""
for example in range(AMOUNT):
	image = np.random.rand(HEIGHT,WIDTH,depth)*255.
	image = np.uint8(np.round(image))
	cv2.imwrite("generated_images/{}.png".format(example),image)
"""
image = np.ones((HEIGHT,WIDTH,depth))*255.
image = np.uint8(np.round(image))
cv2.imwrite("generated_images/1.png",image)
gen_pictures=glob.glob("generated_images/*g")


for sub_f in range(len(sub_folders)):
	folder = "data_set/" + sub_folders[sub_f] + "/"
	shift = det_shift(sub_f)
	for ex_nb in range(50):
		rand_height, rand_width = np.random.randint(40),  np.random.randint(40)
		new_image = image.copy()
		h_min, h_max = rand_height+shift[0]*50, rand_height+shift[0]*50+10
		w_min, w_max = rand_width +shift[1]*50, rand_width +shift[1]*50+10
		new_image[h_min:h_max, w_min:w_max, 1:] = 0
		new_image = np.uint8(new_image)
		cv2.imwrite(folder+str(ex_nb)+".png",new_image)
		print ex_nb, h_min, h_max, w_min, w_max

