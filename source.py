import cv2
import os
import numpy as np
from classifier import LeNet
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import argparse

# Construct argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1)
ap.add_argument("-l", "--load-model", type=int, default=-1)
ap.add_argument("-w", "--weights", type=str)
args = vars(ap.parse_args())

path = 'att_faces'

num_of_file = 0

image_folders = [os.path.join(path, f) for f in os.listdir(path)]

for image_folder in image_folders:
	for image_file in enumerate([os.path.join(image_folder, f) for f in os.listdir(image_folder)]):
		num_of_file += 1

print(num_of_file)

labels = np.zeros(num_of_file, dtype='uint8')
index = 0
keras_in = np.zeros((num_of_file, 1, 112, 92))
for i, image_folder in enumerate(image_folders):
	print(image_folder)
	for j, image_file in enumerate([os.path.join(image_folder, f) for f in os.listdir(image_folder)]):
		#print(image_file)
		labels[index] = i
		image = cv2.imread(image_file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = image / 255.
		image = image[np.newaxis, :, :]
		keras_in[index] = image
		index += 1

print(index)
labels = np_utils.to_categorical(labels, 40)

# Initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01) # learning rate = 0.01
model = LeNet.build(width=92, height=112, depth=1, classes=40, weightsPath=None)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Add checkpoint in case the program quits unexpectedly
checkpoint = ModelCheckpoint(args['weights'], monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print("[INFO] training...")
model.fit(keras_in, labels, validation_split=0.3, batch_size=10, nb_epoch=50, verbose=1, callbacks=callbacks_list)

print("[INFO] dumping weights to file...")
model.save_weights(args["weights"], overwrite=True)